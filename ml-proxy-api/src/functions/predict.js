const { app } = require('@azure/functions');

app.http('predict', {
    methods: ['POST'],
    authLevel: 'anonymous',
    handler: async (request, context) => {
        context.log('Proxying request to SpellSpark ML endpoint...');

        try {
            // 1. Receive the JSON body sent from your Vue app
            const incomingData = await request.json();

            // 2. Azure ML Endpoint details from your test script
            const mlEndpointUrl = 'https://spellspark-recommender.eastus2.inference.ml.azure.com/score';
            
            // Replace with your actual API token (or read from environment variables)
            const apiToken = process.env.AZURE_ML_API_KEY;
            if (!apiToken) {
                throw new Error("AZURE_ML_API_KEY is not defined in environment variables.");
            }
            
            // 3. Make the request to Azure ML (matching your Python script's headers and body)
            const mlResponse = await fetch(mlEndpointUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                    'Authorization': `Bearer ${apiToken}`
                },
                body: JSON.stringify(incomingData)
            });

            // 4. Handle non-200 responses from Azure ML
            if (!mlResponse.ok) {
                const errorText = await mlResponse.text();
                context.log.error(`Azure ML error status: ${mlResponse.status}`);
                return {
                    status: mlResponse.status,
                    jsonBody: { error: 'Azure ML endpoint failed', details: errorText }
                };
            }

            // 5. Parse and return the ML result back to Vue
            const mlResult = await mlResponse.json();
            return {
                status: 200,
                jsonBody: mlResult
            };

        } catch (error) {
            context.log.error('Proxy Error:', error.message);
            return {
                status: 500,
                jsonBody: { error: 'Internal Server Error', message: error.message }
            };
        }
    }
});