# ⚡ SpellSpark: The Spelling Adventure

SpellSpark is a gamified, interactive spelling application built with Vue.js designed to help students master their curriculum-required vocabulary lists. 

Rather than just moving down a static list of words, SpellSpark uses an **Azure Machine Learning (ML) endpoint** to dynamically adapt to the student's needs. If a student struggles with a word, the app seamlessly detours to a smart recommendation tailored to their mistake before returning them to their primary curriculum list.

## ✨ Core Features
* **Curriculum-Driven Learning:** Loads the required vocabulary list (alphabetically) from a local CSV file.
* **Smart ML Recommendations:** When a spelling mistake is made, the app fetches context-aware word recommendations from an Azure ML model. 
* **Seamless Audio Integration:** Uses the browser's native SpeechSynthesis API to auto-play words and provide auditory feedback (applause for success, a friendly "moo" for mistakes).
* **Engaging UI:** Features a kid-friendly cartoon interface with dynamic typing particles and immersive animations.

## 🧠 How the ML Recommendation Engine Works
To ensure security and bypass browser CORS (Cross-Origin Resource Sharing) restrictions, the Vue frontend does not talk to Azure directly. Instead, it uses a local **Azure Function Proxy**.

**The Learning Flow:**
1. The student is presented with a word from the standard CSV curriculum list.
2. **If Correct:** They move on to the next alphabetical word in the list.
3. **If Incorrect:** 
   * The Vue app quietly sends the target word and the misspelled attempt to the local proxy.
   * The proxy securely attaches the Azure API token and queries the deployed Azure ML endpoint.
   * The ML model returns a list of highly similar words based on the student's specific mistake.
   * The app temporarily pauses the main list to serve the top recommended word.
4. Once the student correctly spells the recommended word, the app resumes the standard curriculum right where they left off!

---

## 🚀 Getting Started

To run this project locally, you need to run **two** separate servers: the Proxy API (to talk to Azure) and the Vue UI. 

### Prerequisites
* [Node.js](https://nodejs.org/) installed.
* [Azure Functions Core Tools](https://learn.microsoft.com/en-us/azure/azure-functions/functions-run-local) installed (if running the proxy locally via Azure Functions).
* Your Azure ML Endpoint URL and API Token.

### Step 1: Start the Azure Function Proxy
The proxy **must** be running first so the UI can successfully fetch recommendations.

1. Open your terminal and navigate to the proxy folder:
   ```bash
   cd ml-proxy-api

```

2. Install the backend dependencies:
```bash
npm install

```


3. Create a `local.settings.json` or `.env` file in this folder and add your Azure ML API token (do not commit this file to GitHub!).
4. Start the proxy server:
```bash
func start 

```


*Note: Ensure the proxy is running on a port (like `http://localhost:7071`) that your Vue app is configured to point to.*

### Step 2: Start the Vue UI

Open a **new** terminal window (leave the proxy running in the first one).

1. Navigate to your main Vue project folder:
```bash
# (Run this in the root UI folder)
npm install

```


2. Start the Vue development server:
```bash
npm run dev

```


3. Open your browser to the local address provided (usually `http://localhost:5173`).

---

## 📝 Customizing the Curriculum

To update the required vocabulary list for your students, simply edit the `public/vocabulary-bank.csv` file.

* The app automatically parses this file on startup.
* It trims invisible characters and ignores empty rows.
* Ensure your CSV has a column header named `word` or `Word`.

## 🛠️ Tech Stack

* **Frontend:** Vue 3 (Composition API), Vite, CSS3 Animations, HTML5 Canvas.
* **Data Parsing:** PapaParse (for CSV processing).
* **Backend / Middleman:** Azure Functions (Node.js).
* **AI / ML:** Azure Machine Learning (Managed Online Endpoint).
