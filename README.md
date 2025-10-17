# YouTube OSM Mapper

This project is a web application built with Next.js, React, and TypeScript, designed to visualize YouTube data on an OpenStreetMap interface.

## Tech Stack

*   **Framework:** [Next.js](https://nextjs.org/)
*   **UI Library:** [React](https://react.dev/)
*   **Language:** [TypeScript](https://www.typescriptlang.org/)
*   **Styling:** [Tailwind CSS](https://tailwindcss.com/)
*   **Linting:** [ESLint](https://eslint.org/)

## Getting Started

To get a local copy up and running, follow these steps.

### Prerequisites

You will need Node.js and a package manager like npm, yarn, or pnpm installed on your machine.

### Installation & Setup

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/kuryudesu/youtube-osm-mapper.git
    ```

2.  **Navigate to the project directory:**
    ```sh
    cd youtube-osm-mapper
    ```

3.  **Install dependencies:**
    ```sh
    npm install
    # or
    # yarn install
    # or
    # pnpm install
    ```

4.  **Run the development server:**
    ```sh
    npm run dev
    ```

5.  Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.
6.  You can start editing the main page by modifying `src/app/page.tsx`. The page auto-updates as you edit the file.
   
7.  **Install python dependencies:**
    ```sh
    pip install -r requirements.txt
    ```
8.  **Run the python server:**
    ```sh
    & YOUR_PATH/youtube-osm-mapper/.venv/Scripts/Activate.ps1
    #uvicorn main:app --reload
    ```

## Available Scripts

In the project directory, you can run the following commands:

*   `npm run dev`: Runs the app in development mode.
*   `npm run build`: Builds the application for production usage.
*   `npm run start`: Starts a Next.js production server.
*   `npm run lint`: Runs ESLint to find and fix problems in your code.

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out the [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
