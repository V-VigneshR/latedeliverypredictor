document.getElementById('predict-form').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent form submission

    // Prepare form data
    const formData = new FormData(this);

    // Send data to the server
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Display the prediction result
        document.getElementById('result').innerText = `Prediction: ${data.prediction}`;

        // Display SHAP values as a bar chart
        var shapValues = data.shap_values;  // Assuming the SHAP values are returned from the backend
        var featureNames = data.feature_names;  // Assuming feature names are returned from the backend

        // Prepare SHAP bar chart data
        var shapTrace = {
            x: shapValues,
            y: featureNames,
            type: 'bar',
            orientation: 'h'
        };

        var shapLayout = {
            title: 'SHAP Values for the Prediction',
            xaxis: { title: 'SHAP Value' },
            yaxis: { title: 'Features' }
        };

        // Plot the SHAP values
        Plotly.newPlot('shap-chart', [shapTrace], shapLayout);

        // Display Feature Importances as a bar chart
        var featureImportances = data.feature_importances;  // Assuming feature importances are returned from backend
        var importanceTrace = {
            x: featureNames,
            y: featureImportances,
            type: 'bar'
        };

        var importanceLayout = {
            title: 'Feature Importances',
            xaxis: { title: 'Features' },
            yaxis: { title: 'Importance' }
        };

        // Plot the Feature Importances
        Plotly.newPlot('importance-chart', [importanceTrace], importanceLayout);
    })
    .catch(error => console.error('Error:', error));
});
