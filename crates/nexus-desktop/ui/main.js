import { invoke } from "@tauri-apps/api/core";

function readInput() {
  return {
    age: parseFloat(document.getElementById("age").value),
    gender: document.getElementById("gender").value,
    cna_events: document.getElementById("cna").value,
    mutations: document.getElementById("mutations").value,
  };
}

function renderPredictions(predictions) {
  const tbody = document.querySelector("#results tbody");
  tbody.innerHTML = "";
  const sample = predictions[0];
  if (!sample) return;
  for (const p of sample.predictions) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${p.cancer_type}</td><td>${(p.probability * 100).toFixed(1)}%</td>`;
    tbody.appendChild(tr);
  }
}

document.getElementById("predict-btn").addEventListener("click", async () => {
  try {
    const resp = await invoke("predict", { input: readInput() });
    renderPredictions(resp.predictions);
  } catch (e) {
    alert("Prediction failed: " + e);
  }
});

document.getElementById("explain-btn").addEventListener("click", async () => {
  try {
    const resp = await invoke("explain", { input: readInput() });
    document.getElementById("shap-chart").innerHTML = resp.svg;
  } catch (e) {
    alert("Explanation failed: " + e);
  }
});
