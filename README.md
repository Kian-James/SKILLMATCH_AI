# SkillMatch AI

SkillMatch AI analyzes a resume (PDF or image), extracts the candidate's skills and certifications, matches them against a job corpus using semantic similarity, and shows the best-fit jobs together with skill gaps and an AI-written explanation.

The system is split into three services, and **each one runs in its own Docker container**:

| Service     | Folder           | Technology                       | Port   |
| ----------- | ---------------- | -------------------------------- | ------ |
| Frontend    | `skillmatch_ai/` | Flutter Web served by nginx      | `8080` |
| Backend API | `backend/`       | FastAPI, SBERT, FAISS, Firestore | `8000` |
| OCR service | `ocr-service/`   | FastAPI, EasyOCR, Poppler        | `8001` |

---

## Run with Docker Compose (pull first, then compose)

> **The fastest way to run SkillMatch AI.** The repository ships a `docker-compose.yml`, and all three images are already published on Docker Hub. You do **not** need to build anything, install Flutter, or add a Firebase key file. Pull the images, then start them with Compose:
>
> ```bash
> git clone <your-repository-url> SKILLMATCH_AI
> cd SKILLMATCH_AI
> echo "HF_TOKEN=your_huggingface_token_here" > .env
> docker compose pull
> docker compose up -d
> ```
>
> Then open <http://localhost:8080> and sign in with the [test account](#14-test-account).

### Step by step

**Step 1: Make sure Docker Desktop is running.**

```bash
docker info
```

If this prints an error about the daemon or `docker.sock`, open the Docker Desktop app and wait until the engine is running. Also make sure ports `8000`, `8001` and `8080` are free.

**Step 2: Get the project (this gives you `docker-compose.yml`).**

```bash
git clone <your-repository-url> SKILLMATCH_AI
cd SKILLMATCH_AI
```

**Step 3: Create the `.env` file** in the same folder as `docker-compose.yml`. Compose reads it automatically.

```env
HF_TOKEN=your_huggingface_token_here
```

The token (from <https://huggingface.co/settings/tokens>) is used by the backend to write the AI "fit reason" text. Without it the app still starts, but that text may not be generated.

**Step 4: Pull the images from Docker Hub.**

```bash
docker compose pull
```

This downloads the three prebuilt images named in `docker-compose.yml`. The repositories are public, so no `docker login` is needed.

| Service     | Image pulled                            |
| ----------- | --------------------------------------- |
| Backend     | `skillmatch/skillmatch-backend:latest`  |
| OCR service | `skillmatch/skillmatch-ocr:latest`      |
| Frontend    | `skillmatch/skillmatch-frontend:latest` |

The download is several GB (the OCR image is the largest), so the first pull takes a while. If it stalls, run the same command again and it resumes.

**Step 5: Start everything with Compose.**

```bash
docker compose up -d
```

`-d` runs the containers in the background. Compose creates a shared network, starts the OCR service first, then the backend, then the frontend. Because the images were already pulled, nothing is built.

**Step 6: Check that it is running.**

```bash
docker compose ps
```

All three services should show `running`. The OCR service needs about a minute after starting to load its model. Then test the endpoints:

```bash
curl -s -o /dev/null -w "backend: %{http_code}\n" http://localhost:8000/docs
curl -s -o /dev/null -w "ocr:     %{http_code}\n" http://localhost:8001/health
```

Both should print `200`.

**Step 7: Open the app.**

| What               | URL                            |
| ------------------ | ------------------------------ |
| Web app            | <http://localhost:8080>        |
| Backend API docs   | <http://localhost:8000/docs>   |
| OCR service health | <http://localhost:8001/health> |

### Compose cheat sheet

| Goal                        | Command                                           |
| --------------------------- | ------------------------------------------------- |
| Download the images         | `docker compose pull`                             |
| Start all services          | `docker compose up -d`                            |
| Show service status         | `docker compose ps`                               |
| Live logs (all services)    | `docker compose logs -f --tail 20`                |
| Live logs (one service)     | `docker compose logs -f backend`                  |
| Restart one service         | `docker compose restart backend`                  |
| Stop (keep containers)      | `docker compose stop`                             |
| Start again after a stop    | `docker compose start`                            |
| Stop and remove containers  | `docker compose down`                             |
| Update to the newest images | `docker compose pull` then `docker compose up -d` |

Run these from the folder that contains `docker-compose.yml`. If `docker compose` is not found, update Docker Desktop or try `docker-compose` (with a hyphen).

### Pull vs. build

The `docker-compose.yml` lists both an `image:` and a `build:` for each service:

- **`image:`** is the Docker Hub name. `docker compose pull` downloads it, and `docker compose up -d` then uses it as is.
- **`build:`** is only used when you build from source (`docker compose up -d --build`). See [section 4](#4-build-from-source-with-docker-compose).

If you only want to run the app, pull and compose is all you need.

---

## Table of Contents

1. [How it works](#1-how-it-works)
2. [Project structure](#2-project-structure)
3. [Requirements](#3-requirements)
4. [Build from source with Docker Compose](#4-build-from-source-with-docker-compose)
5. [Running the containers one by one (`docker run`)](#5-running-the-containers-one-by-one-docker-run)
6. [Docker Hub images](#6-docker-hub-images)
7. [Everyday Docker commands](#7-everyday-docker-commands)
8. [Running without Docker (development)](#8-running-without-docker-development)
9. [Configuration](#9-configuration)
10. [API endpoints](#10-api-endpoints)
11. [Deployment on Render](#11-deployment-on-render)
12. [Troubleshooting](#12-troubleshooting)
13. [Security notes](#13-security-notes)
14. [Test account](#14-test-account)

---

## 1. How it works

```
Flutter Web  ──►  Backend (FastAPI)  ──►  OCR service (EasyOCR)
 (port 8080)        (port 8000)             (port 8001)
                        │
                        ├──► SBERT embeddings + FAISS similarity search
                        ├──► Hugging Face LLM (fit-reason text only)
                        └──► Firebase Firestore (jobs + saved results)
```

1. The user signs in (Firebase Auth) and uploads a resume in the Flutter app.
2. The backend extracts text with PyMuPDF. For scanned files it calls the **OCR service**.
3. SBERT semantic matching discovers skills and certifications in the text.
4. FAISS finds the closest jobs, and the backend scores each match.
5. A Hugging Face model writes a short "why this job fits" explanation.
6. Results are saved to Firestore and returned to the app.

---

## 2. Project structure

```
SKILLMATCH_AI/
├── docker-compose.yml          # starts all three services
├── .env                        # secrets for Compose (not committed)
│
├── backend/                    # FastAPI backend
│   ├── Dockerfile
│   ├── .dockerignore
│   ├── app/main.py             # API entry point
│   ├── core/                   # extraction, scoring, validation
│   ├── db/                     # Firestore access
│   ├── pipeline.py
│   ├── config.py               # all environment variables
│   └── requirements.txt
│
├── ocr-service/                # OCR microservice
│   ├── Dockerfile
│   ├── .dockerignore
│   ├── main.py
│   └── requirements.txt
│
└── skillmatch_ai/              # Flutter app (web)
    ├── Dockerfile
    ├── nginx.conf
    ├── .dockerignore
    ├── pubspec.yaml
    └── lib/
```

---

## 3. Requirements

**For Docker (recommended):**

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (includes Docker Compose)
- [Flutter SDK](https://docs.flutter.dev/get-started/install) (only to compile the web app once)
- A Firebase project with Firestore enabled
- About **10 GB of free disk space** and **6 to 8 GB of memory** assigned to Docker
- An internet connection (the backend talks to Firestore and Hugging Face)

Check that Docker works:

```bash
docker --version
docker compose version
docker info
```

If `docker info` prints an error about the daemon or `docker.sock`, open the **Docker Desktop** app and wait until it says the engine is running.

**Give Docker enough memory:** Docker Desktop → **Settings → Resources** → set **Memory** to at least 6 GB (8 GB is better) → **Apply & restart**. EasyOCR and PyTorch need it, and containers can be killed with exit code `137` if memory is too low.

---

## 4. Build from source with Docker Compose

This builds all three images from the source code and then starts them. Use it if you changed the code or want to build the images yourself. If you only want to run the app, use the faster [pull first, then compose](#run-with-docker-compose-pull-first-then-compose) method at the top.

### Step 1: Get the code

```bash
git clone <your-repository-url> SKILLMATCH_AI
cd SKILLMATCH_AI
```

### Step 2: Add the Firebase service account key

The backend needs a Firebase service account key to read and write Firestore.

1. Open the Firebase Console → **Project settings → Service accounts → Generate new private key**.
2. Save the downloaded file as exactly this name inside the `backend/` folder:

   ```
   backend/firebase_service_account.json
   ```

3. Open `backend/.dockerignore` and make sure these lines are **not** present, otherwise the key will not be copied into the image:

   ```
   firebase*.json
   *service_account*.json
   ```

4. Test the key (optional, recommended). It prints `TOKEN OK` when Google accepts the key:

   ```bash
   python3 -m venv /tmp/gtest
   /tmp/gtest/bin/pip install -q google-auth requests
   /tmp/gtest/bin/python -c "
   from google.oauth2 import service_account
   import google.auth.transport.requests as r
   c = service_account.Credentials.from_service_account_file('backend/firebase_service_account.json', scopes=['https://www.googleapis.com/auth/datastore'])
   c.refresh(r.Request())
   print('TOKEN OK')
   "
   ```

   If this prints `invalid_grant: Invalid JWT Signature`, the key is revoked or damaged. Generate a new one.

> Prefer not to bake the key into the image? See [Mounting the key at runtime](#option-b-mount-the-key-at-runtime).

### Step 3: Create the `.env` file

In the project root, create a file named `.env`:

```env
HF_TOKEN=your_huggingface_token_here
```

The token is used by the backend to generate the AI "fit reason" text. Get one at <https://huggingface.co/settings/tokens>.

### Step 4: Create the frontend Docker files

Inside `skillmatch_ai/`, create these three files.

**`skillmatch_ai/Dockerfile`**

```dockerfile
FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY build/web /usr/share/nginx/html
EXPOSE 80
```

**`skillmatch_ai/nginx.conf`**

```nginx
server {
    listen 80;
    root /usr/share/nginx/html;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

**`skillmatch_ai/.dockerignore`** (do **not** ignore `build/`)

```
.dart_tool/
.git/
.idea/
.vscode/
.DS_Store
ios/
android/
macos/
linux/
windows/
```

### Step 5: Compile the Flutter web app

```bash
cd skillmatch_ai
flutter pub get
flutter build web --release
cd ..
```

Check that `skillmatch_ai/build/web/index.html` exists.

> The backend address is set in `skillmatch_ai/lib/services/api_service.dart` (`baseUrl`). For local Docker use it must be `http://localhost:8000`. It is compiled into the app, so **rebuild the web app and the frontend image whenever you change it**.

### Step 6: Create `docker-compose.yml`

In the project root:

```yaml
services:
  ocr-service:
    build: ./ocr-service
    image: skillmatch/skillmatch-ocr:latest
    container_name: ocr-service
    ports:
      - "8001:8001"
    restart: unless-stopped

  backend:
    build: ./backend
    image: skillmatch/skillmatch-backend:latest
    container_name: backend
    ports:
      - "8000:8000"
    environment:
      - PORT=8000
      - OCR_SERVICE_URL=http://ocr-service:8001
      - HF_TOKEN=${HF_TOKEN:-}
    depends_on:
      - ocr-service
    restart: unless-stopped

  frontend:
    build: ./skillmatch_ai
    image: skillmatch/skillmatch-frontend:latest
    container_name: frontend
    ports:
      - "8080:80"
    depends_on:
      - backend
    restart: unless-stopped
```

Inside the Compose network the backend reaches OCR by its service name, `http://ocr-service:8001`.

### Step 7: Remove old containers (only if you ran them before)

Docker will not start a container whose name or port is already taken:

```bash
docker rm -f backend ocr-service frontend
```

### Step 8: Build and start everything

```bash
docker compose up -d --build
```

The first build takes **10 to 20 minutes** because it downloads PyTorch, the SBERT model and the EasyOCR model. Later builds are much faster thanks to caching.

### Step 9: Check that it is running

```bash
docker compose ps
```

All three services should show `running`. Then test the endpoints:

```bash
curl -s -o /dev/null -w "backend: %{http_code}\n" http://localhost:8000/docs
curl -s -o /dev/null -w "ocr:     %{http_code}\n" http://localhost:8001/health
```

Both should print `200`. The OCR service needs about a minute after starting to load its model.

### Step 10: Open the app

| What               | URL                            |
| ------------------ | ------------------------------ |
| Web app            | <http://localhost:8080>        |
| Backend API docs   | <http://localhost:8000/docs>   |
| OCR service health | <http://localhost:8001/health> |

### Step 11: Watch the live logs

```bash
docker compose logs -f --tail 20
```

Press `Ctrl+C` to stop watching. The containers keep running. See [Everyday Docker commands](#7-everyday-docker-commands) for more.

### Step 12: Stop everything

```bash
docker compose stop      # stop, keep containers
docker compose down      # stop and remove containers
```

---

### Seeding the jobs collection (first time only)

If the backend answers with `No jobs found in Firestore collection 'jobs'`, the collection is empty. Seed it once:

```bash
docker compose exec backend python generate_jobs.py
```

### Option B: mount the key at runtime

Instead of copying the key into the image, keep the file outside and mount it when the container starts. The image then contains no secret.

1. Keep `firebase*.json` in `backend/.dockerignore`.
2. In `docker-compose.yml`, add this under the `backend` service:

   ```yaml
   volumes:
     - ./backend/firebase_service_account.json:/app/firebase_service_account.json:ro
   ```

3. Run `docker compose up -d --build`.

---

## 5. Running the containers one by one (`docker run`)

Use this if you do not want Compose. Start the OCR service first.

### Step 1: Build the images

```bash
cd ocr-service
docker build -t ocr-service .

cd ../backend
docker build -t skillmatch-backend .

cd ../skillmatch_ai
docker build -t skillmatch-frontend .
cd ..
```

> Remember the `.` at the end of every `docker build`. It means "use the current folder".

### Step 2: Run them

```bash
# 1. OCR service
docker run -d --name ocr-service -p 8001:8001 ocr-service

# 2. Backend (key baked into the image)
docker run -d --name backend -p 8000:8000 -e PORT=8000 \
  -e OCR_SERVICE_URL=http://host.docker.internal:8001 \
  -e HF_TOKEN=your_huggingface_token_here \
  skillmatch-backend

# 3. Frontend
docker run -d --name frontend -p 8080:80 skillmatch-frontend
```

Why `host.docker.internal`? Inside a container, `localhost` means the container itself, not your computer. `host.docker.internal` is Docker Desktop's name for your computer, where port 8001 is published.

To mount the key at runtime instead, run the backend from the `backend/` folder with:

```bash
docker run -d --name backend -p 8000:8000 -e PORT=8000 \
  -e OCR_SERVICE_URL=http://host.docker.internal:8001 \
  -e HF_TOKEN=your_huggingface_token_here \
  -v "$PWD/firebase_service_account.json:/app/firebase_service_account.json:ro" \
  skillmatch-backend
```

### Step 3: Start and stop later

The containers remember their settings:

```bash
docker start ocr-service backend frontend
docker stop frontend backend ocr-service
```

You only need `docker run` again after you rebuild an image or change settings, ports or environment variables. Remove the old container first with `docker rm -f <name>`.

---

## 6. Docker Hub images

Prebuilt images are published on Docker Hub:

| Service     | Image                            | Link                                                      |
| ----------- | -------------------------------- | --------------------------------------------------------- |
| Backend     | `skillmatch/skillmatch-backend`  | <https://hub.docker.com/r/skillmatch/skillmatch-backend>  |
| OCR service | `skillmatch/skillmatch-ocr`      | <https://hub.docker.com/r/skillmatch/skillmatch-ocr>      |
| Frontend    | `skillmatch/skillmatch-frontend` | <https://hub.docker.com/r/skillmatch/skillmatch-frontend> |

### Run from Docker Hub (no build needed)

The easiest way is [`docker compose pull` and `docker compose up -d`](#run-with-docker-compose-pull-first-then-compose). To start the containers one by one instead:

```bash
docker pull skillmatch/skillmatch-ocr:latest
docker pull skillmatch/skillmatch-backend:latest
docker pull skillmatch/skillmatch-frontend:latest

docker run -d --name ocr-service -p 8001:8001 skillmatch/skillmatch-ocr:latest

docker run -d --name backend -p 8000:8000 -e PORT=8000 \
  -e OCR_SERVICE_URL=http://host.docker.internal:8001 \
  -e HF_TOKEN=your_huggingface_token_here \
  skillmatch/skillmatch-backend:latest

docker run -d --name frontend -p 8080:80 skillmatch/skillmatch-frontend:latest
```

Then open <http://localhost:8080>.

### Push your own build to Docker Hub

1. Log in (use an access token as the password if two-factor authentication is on):

   ```bash
   docker login
   ```

2. With Compose, build and push all three at once. The `image:` names in `docker-compose.yml` decide where each one goes:

   ```bash
   docker compose build
   docker compose push
   ```

3. Or push one image manually. `docker tag` gives your local image its Docker Hub name:

   ```bash
   docker tag skillmatch-backend skillmatch/skillmatch-backend:latest
   docker push skillmatch/skillmatch-backend:latest
   ```

4. Verify on Docker Hub: open the repository, go to the **Tags** tab, and check that `latest` shows a fresh push time.

Notes:

- Pushing again to the same name and tag **updates** the existing repository. It does not create a new one.
- Deleting containers or local images never deletes anything on Docker Hub.
- A locally built image (`skillmatch-backend`) and the pulled image (`skillmatch/skillmatch-backend`) are different images until you `docker tag` the build with the Hub name.
- Do not run `docker pull` on a name before pushing it, because that overwrites your local tag with the old Hub copy.
- Large images take a while. If a push stalls, run the same command again and it resumes.

---

## 7. Everyday Docker commands

### Live logs

```bash
docker logs -f --tail 20 backend
docker logs -f --tail 20 ocr-service
docker logs -f --tail 20 frontend
```

`-f` follows new lines, `--tail 20` starts from the last 20 lines. Press `Ctrl+C` to stop watching (the container keeps running).

All three in one window with name prefixes:

```bash
(docker logs -f --tail 5 backend 2>&1 | sed 's/^/[backend] /' &
 docker logs -f --tail 5 ocr-service 2>&1 | sed 's/^/[ocr] /' &
 docker logs -f --tail 5 frontend 2>&1 | sed 's/^/[frontend] /' &
 wait)
```

With Compose:

```bash
docker compose logs -f --tail 20          # all services
docker compose logs -f backend            # one service
```

More options:

```bash
docker logs --since 5m backend            # last 5 minutes
docker logs -f -t --tail 20 backend       # with timestamps
```

### Status

```bash
docker ps                # running containers
docker ps -a             # all containers, including stopped
docker images            # local images
docker compose ps        # Compose services
```

### Start, stop, restart, remove

```bash
docker start ocr-service backend frontend
docker stop backend ocr-service frontend
docker restart backend
docker rm -f backend                      # remove a container

docker compose up -d --build              # build and start all
docker compose up -d --build backend      # rebuild and recreate one service
docker compose restart backend
docker compose stop
docker compose down
```

### After you change code

Code is copied into the image at build time, so a restart is **not** enough:

```bash
docker compose up -d --build backend
```

Without Compose:

```bash
cd backend
docker build -t skillmatch-backend .
docker rm -f backend
docker run -d --name backend -p 8000:8000 -e PORT=8000 \
  -e OCR_SERVICE_URL=http://host.docker.internal:8001 \
  skillmatch-backend
```

### Clean up

```bash
docker container prune                    # remove all stopped containers
docker image prune -f                     # remove untagged leftover images
docker builder prune -f                   # clear the build cache
```

These do not delete your running containers, your named images, or anything on Docker Hub.

### Look inside a container

```bash
docker exec -it backend sh                # open a shell
docker run --rm skillmatch-backend ls -a /app     # list files inside the image
```

---

## 8. Running without Docker (development)

### OCR service

```bash
cd ocr-service
python3 -m venv venv
source venv/bin/activate
brew install poppler                       # macOS (Linux: apt install poppler-utils)
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001
```

### Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=your_huggingface_token_here
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The key file `backend/firebase_service_account.json` must exist, and the OCR service must be running on port 8001 (the default `OCR_SERVICE_URL` is `http://localhost:8001`).

### Frontend

```bash
cd skillmatch_ai
flutter pub get
flutter run -d chrome
```

Optionally use a fixed port so CORS and Google sign-in see a stable address:

```bash
flutter run -d web-server --web-port 8080
```

Then open <http://localhost:8080>.

To run Flutter in Microsoft Edge on macOS:

```bash
CHROME_EXECUTABLE="/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge" flutter run -d chrome
```

---

## 9. Configuration

### Backend environment variables

| Variable                       | Default                                       | Description                                                        |
| ------------------------------ | --------------------------------------------- | ------------------------------------------------------------------ |
| `PORT`                         | `8000`                                        | Port the API listens on                                            |
| `OCR_SERVICE_URL`              | `http://localhost:8001`                       | Address of the OCR service                                         |
| `OCR_SERVICE_TIMEOUT`          | `150`                                         | Seconds to wait for the OCR service                                |
| `HF_TOKEN`                     | _(empty)_                                     | Hugging Face token for the fit-reason LLM                          |
| `HF_MODEL`                     | `HuggingFaceH4/zephyr-7b-beta:featherless-ai` | Hugging Face model used for fit reasons                            |
| `FIREBASE_CREDENTIALS_PATH`    | `./firebase_service_account.json`             | Path to the service account file                                   |
| `FIREBASE_CREDENTIALS_JSON`    | _(empty)_                                     | Single-line key JSON (used on Render, takes the place of the file) |
| `FIRESTORE_JOBS_COLLECTION`    | `jobs`                                        | Collection with job documents                                      |
| `FIRESTORE_RESULTS_COLLECTION` | `resume_results`                              | Collection where results are saved                                 |
| `TOP_K`                        | `5`                                           | Number of job matches returned                                     |
| `EASYOCR_LANGUAGES`            | `en`                                          | Comma-separated OCR languages                                      |
| `ALLOWED_ORIGINS`              | _(empty)_                                     | Extra comma-separated CORS origins                                 |

Allowed browser origins by default: `http://localhost:8080`, `http://localhost:5000` and the deployed Render frontend. Serve the web app on one of these ports or add yours with `ALLOWED_ORIGINS`.

### OCR service environment variables

| Variable            | Default | Description                   |
| ------------------- | ------- | ----------------------------- |
| `PORT`              | `8001`  | Port the service listens on   |
| `EASYOCR_LANGUAGES` | `en`    | Comma-separated OCR languages |

### Frontend

The backend address is the `baseUrl` constant in `skillmatch_ai/lib/services/api_service.dart`:

| Where the app runs                                          | `baseUrl`                                                                |
| ----------------------------------------------------------- | ------------------------------------------------------------------------ |
| Web, macOS app, iOS simulator (Docker on the same computer) | `http://localhost:8000`                                                  |
| Android emulator                                            | `http://10.0.2.2:8000`                                                   |
| Phone on the same Wi-Fi or hotspot                          | `http://<your-computer-IP>:8000` (find it with `ipconfig getifaddr en0`) |
| Deployed                                                    | `https://<your-backend>.onrender.com`                                    |

Phones also need the computer's firewall to allow incoming connections, and plain `http://` must be enabled in the Android and iOS project settings.

---

## 10. API endpoints

Interactive documentation: <http://localhost:8000/docs>

**Backend (port 8000)**

| Method | Path                      | Purpose                                   |
| ------ | ------------------------- | ----------------------------------------- |
| `POST` | `/thesis-full-analysis`   | Upload a resume and run the full analysis |
| `POST` | `/target-career-analysis` | Analyze a resume against a target career  |
| `POST` | `/refresh-fit-reason`     | Regenerate the AI fit explanation         |

**OCR service (port 8001)**

| Method | Path            | Purpose                          |
| ------ | --------------- | -------------------------------- |
| `GET`  | `/health`       | Health check                     |
| `POST` | `/extract-text` | Extract text from a PDF or image |

---

## 11. Deployment on Render

Each service is deployed as its own Render web service using the **Docker** runtime. Render clones the repository, builds the `Dockerfile` itself, and runs the container. Docker Hub is not involved.

1. In Render, click **New → Web Service** and connect the repository.
2. Set **Root Directory** to `backend` (then repeat for `ocr-service`).
3. Set **Runtime** to **Docker**. Check this under **Settings**. If a `render.yaml` says `env: python`, change it to `runtime: docker`, or the service will not use Docker.
4. Add environment variables:
   - Backend: `HF_TOKEN`, `OCR_SERVICE_URL` (the OCR service's `https://...onrender.com` address), and `FIREBASE_CREDENTIALS_JSON` or a secret file for the key.
   - OCR service: `EASYOCR_LANGUAGES=en`.
5. Deploy. Check the build log for `Building Dockerfile` as proof that Docker is used.
6. Set `baseUrl` in the Flutter app to the backend's Render address, rebuild the web app, and deploy it.

---

## 12. Troubleshooting

| Problem                                               | Cause and fix                                                                                                                                                                           |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `failed to connect to the docker API ... docker.sock` | Docker Desktop is not running. Open it and wait for the engine to start (`docker info` works).                                                                                          |
| `docker build` says it requires 1 argument            | Add the folder at the end: `docker build -t name .`                                                                                                                                     |
| `container name "/backend" is already in use`         | `docker rm -f backend`, then run again (or `docker start -a backend`).                                                                                                                  |
| `port is already allocated`                           | An old container holds the port. Run `docker ps -a` and remove it.                                                                                                                      |
| Backend log: `invalid_grant: Invalid JWT Signature`   | The Firebase key is revoked, disabled or damaged. Generate a new one, replace `backend/firebase_service_account.json`, run the `TOKEN OK` test, then rebuild or recreate the container. |
| Backend log: `DEADLINE_EXCEEDED` from Firestore       | Slow or unstable internet (for example a weak hotspot). Switch to a stronger connection and try again.                                                                                  |
| `ModuleNotFoundError` in backend log                  | A file the app needs is excluded by `.dockerignore`. Remove that line and rebuild.                                                                                                      |
| OCR container exits with code `137`                   | Out of memory. Raise Docker's memory to 6 to 8 GB (Settings → Resources).                                                                                                               |
| Backend cannot reach the OCR service                  | With Compose use `http://ocr-service:8001`. With `docker run` use `http://host.docker.internal:8001`. Check `docker logs ocr-service`.                                                  |
| First analysis takes a long time                      | Models load into memory on first use, and CPU-only inference is slower than the UI estimate. Watch `docker logs -f backend`.                                                            |
| Browser shows a CORS error                            | Serve the web app on port `8080` or `5000`, or add your origin with `ALLOWED_ORIGINS`.                                                                                                  |
| Google sign-in fails on the web                       | Add `http://localhost:8080` under **Authorized JavaScript origins** in the Google Cloud OAuth client.                                                                                   |
| `flutter run` says `No client connected`              | The browser tab was closed or lost its connection. Press `q` and run again. Do not open the printed URL in a second tab.                                                                |
| Frontend image build cannot reach `ghcr.io`           | The network blocks it. Use the local build in Step 5 (`flutter build web`) with the `nginx:alpine` Dockerfile.                                                                          |
| Frontend page is blank or old                         | Rebuild with `flutter build web --release`, then `docker compose up -d --build frontend`.                                                                                               |
| `docker compose` not found                            | Update Docker Desktop, or try `docker-compose` (with a hyphen).                                                                                                                         |

---

## 13. Security notes

- **The backend image contains a Firebase service account key** when you build it with the key in `backend/`. Anyone who can pull that image can read it and access your Firestore data. If the image is on a public Docker Hub repository, treat the key as public:
  - Create a dedicated service account with only the **Cloud Datastore User** role for this project.
  - Do not keep real personal data (real resumes) in that Firestore while the key is public.
  - Delete the key in Google Cloud (**IAM & Admin → Service Accounts → Keys**) when you no longer need it, and generate a new one when you do.
- For a safer setup, use [Option B](#option-b-mount-the-key-at-runtime) so the image holds no secret.
- Never commit `.env`, `firebase_service_account.json` or any other key to Git. Add them to `.gitignore`.
- Keep real resumes out of images. The OCR `.dockerignore` excludes `temp_resumes/`, and the backend one excludes `resumes/`.
- Docker Hub keeps old image layers downloadable by digest even after a new push. If a secret was ever pushed, revoke the key in Google Cloud, because overwriting `latest` does not undo the exposure.

---

## 14. Test account

Use this account to try the app without registering:

| Field    | Value                 |
| -------- | --------------------- |
| Email    | `test@skillmatch.com` |
| Password | `TestPassword123!`    |

**How to sign in**

1. Open the app at <http://localhost:8080> (or the deployed link).
2. On the login screen, enter the email and password above and sign in.
3. Upload a resume (PDF or image) to run an analysis.

**Guest login:** the login screen also has a guest option, which signs in anonymously. Guest scan history is lost when you exit the guest session.

**If sign-in fails:** check that the Email/Password and Anonymous sign-in methods are enabled in the Firebase Console (**Authentication → Sign-in method**), and that the account exists under **Authentication → Users** in the same Firebase project the app is configured for.

> This is a throwaway account for demonstration only. Do not store real personal data in it.

---

## Authors

- James, Kian Andrei G.
- Mendoza, Gerail C.
- Halili, Jero E.
- Tinio, Angelica Z.

## License

Copyright (c) 2026 James, Kian Andrei G.; Mendoza, Gerail C.; Halili, Jero E.; Tinio, Angelica Z. All rights reserved.
