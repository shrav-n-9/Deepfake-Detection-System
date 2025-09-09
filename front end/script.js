const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const resultDiv = document.getElementById("result");

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;

  // Show preview
  preview.src = URL.createObjectURL(file);
  preview.style.display = "block";

  // Prepare request
  const formData = new FormData();
  formData.append("file", file);

  resultDiv.className = "result";
  resultDiv.classList.remove("hidden");
  resultDiv.innerHTML = "⏳ Analyzing...";

  try {
    const res = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formData
    });
    const data = await res.json();

    if (data.error) {
      resultDiv.innerHTML = "❌ Error: " + data.error;
      return;
    }

    const predClass = data.prediction === "fake" ? "fake" : "real";
    const emoji = data.prediction === "fake" ? "🤖" : "✅";

    resultDiv.className = "result " + predClass;
    resultDiv.innerHTML = `${emoji} <b>${data.prediction.toUpperCase()}</b><br>Confidence: ${(data.confidence*100).toFixed(1)}%`;
  } catch (err) {
    resultDiv.innerHTML = "⚠️ Server error: " + err.message;
  }
});
