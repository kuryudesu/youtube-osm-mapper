import os
import logging
import tempfile
import re
import textwrap
import json
import asyncio
import threading
import subprocess
import sys
from contextlib import asynccontextmanager

import uvicorn
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api import NoTranscriptFound, TranscriptsDisabled
# from youtube_transcript_api.proxies import WebshareProxyConfig
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError
from google.api_core import exceptions as google_exceptions
from dotenv import load_dotenv
from typing import List, Dict, Optional

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# --- Pydantic Schemas (like Zod in TS) ---
class Location(BaseModel):
    name: str = Field(..., description="The name of the location.")
    description: str = Field(..., description="A brief description of the location based on the video's context.")
    lat: float = Field(..., description="The estimated latitude.")
    lng: float = Field(..., description="The estimated longitude.")

class ApiResponse(BaseModel):
    locations: List[Location] = Field(description="A list of extracted locations from the video.")
    transcript: Optional[str] = Field(None, description="The full transcript of the video, which was used for analysis.")

class UrlRequest(BaseModel):
    url: str
    apiKey: str
    languageCode: Optional[str] = None
    aiLanguageCode: Optional[str] = None

class LanguageRequest(BaseModel):
    url: str

class Language(BaseModel):
    language: str
    language_code: str


# --- In-memory Cache ---
# A simple dictionary to cache results for processed video IDs.
# Key: video_id (str), Value: ApiResponse
processing_cache: Dict[str, ApiResponse] = {}
# Cache for transcripts to avoid re-downloading/re-transcribing on retries.
transcript_cache: Dict[str, str] = {}
# Cache for downloaded audio bytes to avoid re-downloading on retries.
audio_cache: Dict[str, bytes] = {}

# --- Global Locks for API Key safety ---
# The google-generativeai library uses a global configuration for the API key.
# To use per-request keys safely in a concurrent server, we must use locks
# to prevent race conditions where one request's key overwrites another's.
gemini_async_lock = asyncio.Lock() # For the main async event loop
gemini_sync_lock = threading.Lock() # For sync functions running in the thread pool

# --- FastAPI App Initialization ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Application starting up.")

    # At startup, check for yt-dlp updates to ensure it's always the latest version.
    # This is crucial as YouTube often changes things that can break downloading.
    def check_and_update_ytdlp():
        try:
            logging.info("Checking for yt-dlp updates...")
            # We use sys.executable to ensure we're using the pip from the current Python environment.
            # This is more robust than just calling 'pip'.
            # The command will update yt-dlp if a newer version is available.
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"],
                capture_output=True, text=True, check=False, timeout=60
            )
            if result.returncode == 0:
                logging.info("yt-dlp update check completed successfully.")
            else:
                logging.error(f"yt-dlp update command failed with exit code {result.returncode}:\n{result.stderr}")
        except Exception as e:
            logging.error(f"An unexpected error occurred during yt-dlp update check: {e}")
    # Run the blocking subprocess call in a thread pool to not block the event loop.
    await run_in_threadpool(check_and_update_ytdlp)
    yield
    # Clean up resources if needed
    logging.info("Shutting down.")

app = FastAPI(lifespan=lifespan)

# --- CORS Middleware ---
# In production, this should be your Vercel deployment URL.
# We build a dynamic list of allowed origins for CORS.
origins = [
    "https://youtube-osm-mapper.vercel.app", # Production frontend
    "http://localhost:3000",                 # Local development frontend
]

# Dynamically add URLs from environment variables for development tunnels.
# This robustly handles URLs with or without a trailing slash.
for env_var in ["DEV_TUNNEL_URL", "FRONTEND_DEV_TUNNEL_URL"]:
    url = os.getenv(env_var)
    if url:
        # Browsers send the Origin header without a trailing slash.
        # We add the base URL and, just in case, the one with a slash.
        origins.append(url.rstrip('/'))
        origins.append(url.rstrip('/') + '/')


app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin for origin in origins if origin], # Filter out None values
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions ---
def extract_video_id(url: str) -> str:
    """
    Extracts the YouTube video ID from various URL formats using a regular expression.
    This is a robust way to handle formats like youtu.be/, /watch, /embed, /shorts,
    and URLs with timestamps or other parameters.
    """
    # This regex covers standard, mobile, short, and embed links.
    regex = r'(?:https?:\/\/)?(?:www\.|m\.)?(?:youtube\.com\/(?:watch\?v=|v\/|embed\/|shorts\/)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(regex, url)
    if match:
        video_id = match.group(1)
        logging.info(f"Extracted video ID: {video_id} from URL: {url}")
        return video_id
    raise ValueError(f"Could not extract a valid YouTube video ID from the URL: {url}")

def get_transcript_from_api(video_url: str, language_codes: List[str]) -> str:
    """Attempts to fetch a pre-generated transcript."""
    video_id = extract_video_id(video_url)
    try:
        ytt_api = YouTubeTranscriptApi()
        # First, get a list of all available transcripts.
        transcript_list_obj = ytt_api.list(video_id)

        logging.info(f"Attempting to fetch transcript for video ID: {video_id} with languages: {language_codes}")
        
        # Then, find the specific transcript we want from the list.
        transcript = transcript_list_obj.find_transcript(language_codes)
        fetched_transcript = transcript.fetch()
        if not fetched_transcript:
            raise ValueError("Empty transcript returned from API.")
        logging.info(f"Successfully fetched language: {transcript.language_code}")
        logging.info(f"Fetched {len(fetched_transcript)} transcript segments.")
        transcript_text = " ".join([t.text for t in fetched_transcript])
        logging.info(f"Transcript for video data: {transcript_text}")
        logging.info("Successfully fetched transcript via API.")
        return transcript_text
    except (NoTranscriptFound, TranscriptsDisabled):
        logging.warning(f"No pre-generated transcript found for {video_id}. Attempting to generate one.")
        try:
            ytt_api = YouTubeTranscriptApi()
            gen_transcript_list = ytt_api.list(video_id)
            for transcript in gen_transcript_list:
            # the Transcript object provides metadata properties
                print(
                    transcript.video_id,
                    transcript.language,
                    transcript.language_code,
                    # whether it has been manually created or generated by YouTube
                    transcript.is_generated,
                    # a list of languages the transcript can be translated to
                    transcript.translation_languages,
                )

            generated_transcript = gen_transcript_list.find_generated_transcript(language_codes)
            if not generated_transcript:
                raise ValueError("Empty transcript returned from generated API.")
            logging.info(f"Successfully generated transcript for language: {generated_transcript.language_code}")
            transcript_text = " ".join([t.text for t in generated_transcript.fetch()])
            return transcript_text
        except Exception as e:
            logging.warning(f"Failed to generate transcript: {e}")
            raise
    except Exception as e:
        logging.warning(f"youtube-transcript-api failed: {e}")
        raise

def transcribe_audio_with_ai(video_url: str, video_id: str, api_key: str, target_language: str) -> str:
    """Fallback: Downloads audio and transcribes it using Gemini."""
    # Acquire a thread-safe lock before modifying the global genai config
    with gemini_sync_lock:
        # This function runs in a separate thread. The lock prevents race conditions
        # with other threads that might also be performing transcriptions.
        genai.configure(api_key=api_key)
        # audio_bytes = None # This line is not needed

        try:
            logging.info("Fallback: Transcribing with AI.")

            # Check audio cache first
            if video_id in audio_cache:
                audio_bytes = audio_cache[video_id]
                logging.info(f"Audio for video ID {video_id} found in cache.")
            else:
                # Use a temporary directory to handle file naming by yt-dlp
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Configure yt-dlp to download and convert to mp3.
                    ydl_opts = {
                        'format': 'bestaudio/best',
                        'outtmpl': os.path.join(tmpdir, '%(id)s.%(ext)s'),
                        'postprocessors': [{
                            'key': 'FFmpegExtractAudio',
                            'preferredcodec': 'mp3',
                            'preferredquality': '128',
                        }],
                        'noplaylist': True,
                        'quiet': True,
                    }
                    # # Add proxy support for yt-dlp as well, as it's likely also blocked.
                    # proxy_url = os.getenv("YTT_PROXY")
                    # if proxy_url:
                    #     logging.info("Using proxy for yt-dlp.")
                    #     ydl_opts['proxy'] = proxy_url

                    logging.info("Downloading and converting audio to MP3 with yt-dlp...")
                    with YoutubeDL(ydl_opts) as ydl:
                        info_dict = ydl.extract_info(video_url, download=True)
                        expected_filename = f"{info_dict['id']}.mp3"
                        downloaded_file_path = os.path.join(tmpdir, expected_filename)

                    if not os.path.exists(downloaded_file_path):
                        raise FileNotFoundError(f"yt-dlp did not produce the expected MP3 file: {expected_filename}")

                    with open(downloaded_file_path, 'rb') as f:
                        audio_bytes = f.read()

                    if not audio_bytes:
                        raise ValueError("Downloaded audio file is empty.")

                    # Store in audio cache for future retries
                    audio_cache[video_id] = audio_bytes
                    logging.info(f"Audio for video ID {video_id} stored in cache.")

                logging.info(f"Audio downloaded and converted ({len(audio_bytes)} bytes). Transcribing {target_language} with Gemini...")

                # Using a faster and more cost-effective model for transcription
                model = genai.GenerativeModel('gemini-2.5-flash')
                
                transcription_prompt = textwrap.dedent(f"""\
                    你是一位專業的逐字稿專家，專門處理包含口語和背景噪音的音訊。
                    你的任務是將以下音訊內容，精確地轉錄成一份乾淨、易讀的「{target_language}」逐字稿。
                    請遵守以下規則：
                    2.  **準確性**: 盡可能保留原始口語內容，包含語氣詞（例如：嗯、啊、啦）和重複的詞語，以確保真實性。
                    1.  **語言**: 輸出內容必須是「{target_language}」!!!。這是最重要的規則。
                    2.  **準確性**: 盡可能保留原始口語內容，包含語氣詞和重複的詞語，以確保真實性。
                    3.  **格式**: 請使用標準的中文標點符號（例如：，、。、？）來斷句，提升可讀性。
                    4.  **簡潔**: 只輸出轉錄後的文字，不要包含任何額外的標題、註解或說明文字（例如，不要寫「這是逐字稿：」）。""")

                response = model.generate_content([
                    transcription_prompt,
                    {'mime_type': 'audio/mpeg', 'data': audio_bytes}
                ])

                generated_text = response.text
                if generated_text.startswith(transcription_prompt):
                    generated_text = generated_text[len(transcription_prompt):].lstrip()

                if not generated_text:
                    raise ValueError("AI transcription returned empty text.")

                logging.info("Successfully transcribed audio with Gemini.")
                return generated_text

        except DownloadError as e:
            logging.error(f"yt-dlp download failed: {e}")
            reason = "the video's audio could not be downloaded. This can happen with age-restricted, private, or live videos."
            raise RuntimeError(f"The standard transcript was unavailable, and the AI transcription fallback failed because {reason}")
        except google_exceptions.GoogleAPICallError as e:
            logging.error(f"Gemini API call failed during transcription: {e}")
            reason = f"the AI model failed to process the audio. Details: {e}"
            raise RuntimeError(f"The standard transcript was unavailable, and the AI transcription fallback failed because {reason}")
        except Exception as e:
            logging.error(f"An unexpected error occurred in AI transcription fallback: {e}", exc_info=True)
            reason = f"an unknown issue occurred. Error: {str(e)}"
            raise RuntimeError(f"The standard transcript was unavailable, and the AI transcription fallback failed because {reason}")
        finally:
            # Clean up the global configuration to avoid side effects.
            genai.configure(api_key=None)


async def stream_video_processing(video_url: str, api_key: str, language_code: Optional[str] = None, ai_language_code: Optional[str] = None):
    """
    An async generator that processes the video and yields status updates as Server-Sent Events.
    """
    try:
        if not api_key:
            raise ValueError("Gemini API key is required.")
        
        # Models can be initialized once. They will pick up the global API key
        # at the time of their method call, which we will set inside a lock.
        json_model = genai.GenerativeModel(
            "gemini-2.5-flash",
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
            )
        )
        
        yield "event: log\ndata: 正在初始化 AI 模型...\n\n"

        # Step 2: Extract video ID and check cache
        video_id = extract_video_id(video_url)
        if video_id in processing_cache:
            yield f"event: log\ndata: 找到快取資料，正在回傳結果...\n\n"
            result_data = processing_cache[video_id].model_dump_json()
            yield f"event: result\ndata: {result_data}\n\n"
            return

        yield f"event: log\ndata: 正在處理新的影片 (ID: {video_id})...\n\n"

        # Step 3: Get transcript, using cache if available
        transcribed_text = ""
        if video_id in transcript_cache:
            transcribed_text = transcript_cache[video_id]
            yield "event: log\ndata: 從快取中找到逐字稿。\n\n"
        else:
            # If a specific language is chosen and it's not the AI fallback
            if language_code and language_code != "ai-zh":
                try:
                    yield f"event: log\ndata: 正在嘗試獲取 {language_code} 語言的逐字稿...\n\n"
                    transcribed_text = await run_in_threadpool(get_transcript_from_api, video_url, [language_code])
                    yield "event: log\ndata: 成功獲取逐字稿。\n\n"
                except Exception as e:
                    sanitized_error = str(e).replace('\n', ' ').replace('\r', '')
                    yield f"event: log\ndata: 無法獲取指定逐字稿 ({sanitized_error})，將改用 AI 進行語音轉文字...\n\n"
                    transcribed_text = await run_in_threadpool(transcribe_audio_with_ai, video_url, video_id, api_key, ai_language_code)
                    yield "event: log\ndata: 成功使用 AI 完成語音轉文字。\n\n"
            else: # Default to AI or if AI was explicitly chosen
                if language_code == "ai-zh":
                    yield "event: log\ndata: 已選擇 AI 語音辨識，開始進行語音轉文字...\n\n"
                else: # This case is for when no language is selected, and we fallback to AI
                    yield "event: log\ndata: 未指定逐字稿語言，將使用 AI 進行語音轉文字...\n\n"
                transcribed_text = await run_in_threadpool(transcribe_audio_with_ai, video_url, video_id, api_key, ai_language_code)
                yield "event: log\ndata: 成功使用 AI 完成語音轉文字。\n\n"

            # Cache the newly obtained transcript for future retries
            if transcribed_text:
                transcript_cache[video_id] = transcribed_text
                logging.info(f"Transcript for video ID {video_id} stored in cache.")
                yield "event: log\ndata: 逐字稿已存入快取，供後續使用。\n\n"

        # Step 4: Extract locations from text using Gemini JSON mode
        yield "event: log\ndata: 正在使用 AI 從逐字稿中提取地點資訊...\n\n"

        api_response_schema = {
            "type": "object",
            "properties": {
                "locations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "lat": {"type": "number"},
                            "lng": {"type": "number"},
                        },
                        "required": ["name", "description", "lat", "lng"],
                    },
                },
            },
            "required": ["locations"],
        }

        prompt = textwrap.dedent(f"""\
            你是一位專業的地理學家和數據分析師，專門從非結構化文本中提取地理空間信息。
            你的任務是仔細分析以下來自YouTube旅遊影片的轉錄文本，並提取其中明確提到的所有地點。
            請嚴格遵守以下規則和輸出格式：
            1.  **輸出格式**:
                *   你必須以 JSON 格式回應，並嚴格遵守提供的 schema。
                *   對於每一個提取出的地點，請提供以下信息：
                    *   `name`: 地點的官方或最常見的名稱。
                    *   `description`: 根據影片文本，簡潔地總結講者在該地點的活動或對該地點的描述。描述應直接與文本內容相關。
                    *   `lat`: 該地點最精確的緯度座標。對於城市等大範圍區域，請提供市中心座標。
                    *   `lng`: 該地點最精確的經度座標。
            2.  **提取規則**:
                *   輸出內容必須是「{ai_language_code}」。
                *   只提取影片中實際參觀或詳細描述的地點。
                *   請勿提取地名（例如，「大阪」、「京都」）。
                *   請勿包含僅僅順帶提及、沒有足夠上下文進行地理定位的地點（例如，只是說「我要去日本」）。
                *   如果文本中沒有提到任何符合條件的地點，你必須返回一個空的 `locations` 數組，像這樣：`{{"locations": []}}`。
            待分析文本如下：
            ---
            "{transcribed_text}"
            ---""")
        
        logging.info("Extracting locations from text...")
        # Acquire an async lock before modifying the global genai config
        async with gemini_async_lock:
            genai.configure(api_key=api_key)
            try:
                response = await json_model.generate_content_async(
                    [prompt],
                    generation_config=genai.GenerationConfig(response_schema=api_response_schema)
                )
            finally:
                # Clean up immediately after the call to minimize the lock time
                genai.configure(api_key=None)

        validated_data = ApiResponse.model_validate_json(response.text)
        validated_data.transcript = transcribed_text

        yield f"event: log\ndata: 成功提取 {len(validated_data.locations)} 個地點！正在完成處理...\n\n"

        # Step 5: Cache and yield final result
        processing_cache[video_id] = validated_data
        result_json = validated_data.model_dump_json()
        yield f"event: result\ndata: {result_json}\n\n"
        logging.info(f"Successfully extracted and validated {len(validated_data.locations)} locations.")

        logging.info(f"Result for video ID {video_id} stored in cache.")
        yield "event: log\ndata: 處理完成！\n\n"

    except Exception as e:
        logging.error(f"Error in processing stream for URL {video_url}: {e}", exc_info=True)
        error_str = str(e)
        if "the video's audio could not be downloaded" in error_str:
            error_payload = json.dumps({"detail": "處理影片時發生錯誤: 你的IP可能被YouTube封鎖，導致無法下載音訊。請嘗試更換網路環境、使用代理(Proxy)或稍後再試。"})
        elif "https://ai.google.dev/gemini-api/docs/rate-limits" in error_str:
            error_payload = json.dumps({"detail": "你已經超過了 Google Gemini API 免費層級的每分鐘請求次數限制，因此被暫時阻擋，必須等一段時間才能再次請求。"})
        else:
            error_payload = json.dumps({"detail": f"處理影片時發生錯誤: {error_str}"})
        yield f"event: error\ndata: {error_payload}\n\n"

# --- API Endpoints ---
@app.post("/api/get-languages")
async def get_languages(request: LanguageRequest):
    """
    Given a YouTube URL, returns a list of available transcript languages.
    """
    try:
        video_id = extract_video_id(request.url)
        # First, check if the full result is already in the processing cache.
        if video_id in processing_cache:
            logging.info(f"Full result for video ID {video_id} found in cache. Skipping language fetch.")
            return {"cached": True, "languages": []}

        # This function runs in a thread pool to avoid blocking the event loop
        ytt_api = YouTubeTranscriptApi()
        transcript_list = await run_in_threadpool(ytt_api.list, video_id)
        languages = [
            Language(language=t.language, language_code=t.language_code)
            for t in transcript_list
        ]
        # Always add the AI option as a fallback or primary choice
        languages.append(Language(language="AI 語音辨識 (推薦)", language_code="ai-zh"))
        
        return {"languages": languages}
    
    except (NoTranscriptFound, TranscriptsDisabled):
        # If no pre-made transcripts exist, only AI is an option
        return {"languages": [Language(language="AI 語音辨識 (唯一選項)", language_code="ai-zh")]}
    except Exception as e:
        logging.error(f"Error fetching languages for {request.url}: {e}", exc_info=True)
        # Instead of raising an HTTP 500 error, return a controlled response.
        # This allows the frontend to display a message while still offering the AI fallback.
        if isinstance(e, TranscriptsDisabled):
            error_message = f"無法獲取語言清單，因為影片未啟用字幕但您仍可使用 AI 語音辨識。"
        else:
            error_message = f"無法獲取語言清單(可能是網路問題)，但您仍可使用 AI 語音辨識。錯誤: {e}"
        return {"languages": [Language(language="AI 語音辨識 (推薦)", language_code="ai-zh")], "error": error_message}



# --- API Endpoint ---
@app.post("/api/extract-locations")
async def extract_locations(request: UrlRequest):
    return StreamingResponse(stream_video_processing(request.url, request.apiKey, request.languageCode, request.aiLanguageCode), media_type="text/event-stream")

# --- To run the server ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
