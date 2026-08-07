# SpellSpark ML Proxy API (`ml-proxy-api`)

An Azure Functions (Node.js v4) proxy server built to securely handle communication between the **`spell-spark-ui`** Vue frontend and the **SpellSpark Azure Machine Learning Recommender Model**.

---

## 🚀 Purpose

Directly connecting a web frontend (such as `spell-spark-ui` running on `http://localhost:5173`) to an Azure ML endpoint causes two major issues:

1. **CORS (Cross-Origin Resource Sharing) Restrictions:** Browsers block direct requests from `localhost` to Azure ML inference endpoints due to missing CORS headers on the Azure ML side.
2. **Security Vulnerabilities:** Hardcoding the Azure ML Bearer token inside frontend JavaScript exposes your secret keys to anyone inspecting the browser network tab.

This proxy service sits between your UI and Azure ML to:
- Act as a backend layer that handles CORS for local development and cloud deployments.
- Securely store and inject the `AZURE_ML_API_KEY` on the server side so secrets are never sent to the browser.

---

## 🛠️ Prerequisites

Before running this project locally, ensure you have:

- **Node.js** (v18 or v20 recommended)
- **Azure Functions Core Tools v4**

On macOS, install Azure Functions Core Tools using Homebrew:

```bash
brew tap azure/functions
brew trust azure/functions
brew install azure-functions-core-tools@4

```

Verify installation:

```bash
func --version

```

---

## ⚙️ Local Configuration (`local.settings.json`)

To run the function locally, you **must create a `local.settings.json` file** in the root directory of this project.

> ⚠️ **Important:** `local.settings.json` contains private credentials and is included in `.gitignore`. **Never commit this file to source control.**

Create `local.settings.json` in the project root and add the following template:

```json
{
  "IsEncrypted": false,
  "Values": {
    "AzureWebJobsStorage": "",
    "FUNCTIONS_WORKER_RUNTIME": "node",
    "AZURE_ML_API_KEY": "YOUR_AZURE_ML_API_KEY_HERE"
  },
  "Host": {
    "CORS": "http://localhost:5173",
    "CORSCredentials": false
  }
}

```

Replace `YOUR_AZURE_ML_API_KEY_HERE` with your actual primary or secondary inference token for the Azure ML endpoint.

---

## 🏃 How to Run Locally

1. **Install Dependencies:**
```bash
npm install

```


2. **Start the Function Runtime:**
```bash
func start

```


3. The function worker will start and expose the proxy endpoint:
* **URL:** `http://localhost:7071/api/predict`
* **Method:** `POST`



---

## 🔄 Integration with `spell-spark-ui`

This proxy is designed to work seamlessly alongside the **`spell-spark-ui`** Vue app.

### Workflow:

1. Start this proxy first using `func start` (runs on `http://localhost:7071`).
2. Start your Vue app in a separate terminal using `npm run dev` inside `spell-spark-ui` (runs on `http://localhost:5173`).
3. Your Vue app sends request payloads to `http://localhost:7071/api/predict`.

### Example Payload Structure:

```json
{
  "requests": [
    {
      "correct_word": "missing",
      "incorrect_word": "mising"
    }
  ]
}

```

---

## ☁️ Deployment to Azure

When deploying this Function App to Azure:

1. Publish the project using the Azure Functions VS Code Extension or Azure CLI (`func azure functionapp publish <YOUR_APP_NAME>`).
2. Go to your **Azure Portal** $\rightarrow$ **Function App** $\rightarrow$ **Settings** $\rightarrow$ **Environment variables**.
3. Add a new Application Setting named `AZURE_ML_API_KEY` containing your ML token.
4. In the Function App **CORS** settings, add your production Vue application URL.