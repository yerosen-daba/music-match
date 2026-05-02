# Music Byy

**Music Byy** (formerly known as Music Match) is a full-stack web application designed to calculate the musical "vibe" compatibility between two users based on their favorite songs. 

Unlike standard matching algorithms that rely heavily on explicit genre tags, Music Byy calculates compatibility by dynamically fetching and comparing metadata across three core acoustic characteristics:
1. **Energy (Popularity/Rank Proxy)**: Evaluates if both users prefer mainstream hits or underground/niche music.
2. **Tempo (BPM)**: Calculates the average tempo of the user's favorite tracks to determine if they prefer upbeat or relaxed music.
3. **Valence (Release Era Proxy)**: Evaluates the release year of the tracks to determine if the users have similar tastes in classic vs. modern music.

After computing a Gaussian mathematical similarity score, the engine fetches related artists from the Deezer API and securely interleaves 6 new song recommendations for the users to enjoy together.

## Setup & Running the Code

The backend is built with **FastAPI** in Python, and the frontend is built using standard **HTML/CSS/Vanilla JavaScript**. The project is designed to be entirely self-contained and **does not require any personal API keys** (it relies on Deezer's public, unauthenticated API).

### Prerequisites
- Python 3.9+
- A modern web browser

### Running Locally
1. Clone the repository to your local machine.
2. Navigate to the project directory:
   ```bash
   cd music-match
   ```
3. Install the lightweight Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the FastAPI backend server:
   ```bash
   uvicorn app:app --reload
   ```
   *The server will start on `http://localhost:8000`.*
5. Open the `index.html` file in any web browser to view the frontend interface. It will automatically connect to your local backend.

### Deployment Architecture
The live version of this app is deployed as follows:
- **Backend**: Hosted as a Dockerized web service on Render (Starter Tier).
- **Frontend**: Hosted statically via GitHub Pages and mapped to a custom domain (`musicbyy.com`).

---

## File Structure

- `app.py`: The main FastAPI server entry point and API router.
- `client.py`: A singleton managing asynchronous HTTP connections (via `httpx`).
- `deezer.py`: Handles all interactions with the Deezer API (fetching track metadata and relationships).
- `match.py`: The core algorithm engine. Contains the complex Gaussian math that scores compatibility and generates the final recommendation list.
- `index.html`: The fully responsive, vanilla frontend user interface.
- `Dockerfile`: Configuration for deploying the backend environment.

---

## Citations & External Contributions

### Code Sources & Dependencies
- **FastAPI Framework**: Used for building the backend REST API quickly and asynchronously ([Link](https://fastapi.tiangolo.com/)).
- **Deezer API**: All track search, artist relationship mapping, and metadata fetching are powered by the public Deezer API ([Link](https://developers.deezer.com/api)).

### Generative AI Usage
I extensively used **Antigravity** (a generative AI pair-programming assistant powered by Google DeepMind) throughout the development of this project to assist with architecture, math optimization, and deployment.

Specifically, the AI was used to:
1. **Refactor the Codebase**: The initial prototype was a single monolithic Python script. I used the AI to help me split the codebase into a clean, modular structure (`app.py`, `deezer.py`, `match.py`) to improve maintainability.
2. **Optimize the Mathematical Engine**: The Gaussian compatibility scoring formula in `match.py` (specifically `compute_compatibility`) was generated in collaboration with the AI. We iterated on the standard deviation (`sigma`) values to correctly scale the differences in BPM and Release Years so that the scores were balanced and realistic.
3. **Deployment Troubleshooting**: The backend originally relied on `librosa` for deep audio waveform analysis. However, downloading and processing MP3 files caused severe "Out of Memory" crashes and 30-second timeouts on the Render cloud host. The AI analyzed the server error logs and helped me rewrite the engine to bypass audio downloading entirely, using metadata proxies (Rank, BPM, Year) instead, which reduced API response times from ~39 seconds to ~1.3 seconds.
4. **CSS Animations**: The complex pulsing animations and dynamic visual layout of the `index.html` frontend were generated using the AI, which provided the vanilla CSS keyframes to achieve the "glassmorphism" design aesthetic.
