const { invoke } = window.__TAURI__.core;

function readInput() {
  return {
    age: parseFloat(document.getElementById("age").value),
    gender: document.getElementById("gender").value,
    cna_events: document.getElementById("cna").value,
    mutations: document.getElementById("mutations").value,
  };
}

function setStatus(ok, text) {
  const pill = document.getElementById("engine-status");
  pill.classList.toggle("offline", !ok);
  pill.innerHTML = `<span class="dot"></span> ${text}`;
}

function setGauge(pct) {
  const gauge = document.getElementById("gauge");
  const value = document.getElementById("gauge-value");
  const caption = document.getElementById("gauge-caption");
  gauge.style.setProperty("--pct", pct.toFixed(1));
  value.textContent = `${pct.toFixed(1)}%`;
  caption.textContent =
    pct >= 80 ? "High-confidence call" : pct >= 50 ? "Moderate confidence" : "Low confidence";
}

function renderPredictions(sample) {
  const list = document.getElementById("results");
  list.innerHTML = "";
  if (!sample || !sample.predictions.length) {
    list.innerHTML = `<li class="empty">No predictions yet.</li>`;
    return;
  }

  sample.predictions.forEach((p, i) => {
    const pct = p.probability * 100;
    const li = document.createElement("li");
    li.className = "bar-row" + (i === 0 ? " top" : "");
    li.innerHTML = `
      <div class="bar-top">
        <span class="bar-name">${p.cancer_type}</span>
        <span class="bar-pct">${pct.toFixed(1)}%</span>
      </div>
      <div class="bar-track"><div class="bar-fill"></div></div>`;
    list.appendChild(li);
    requestAnimationFrame(() => {
      li.querySelector(".bar-fill").style.width = `${pct}%`;
    });
  });

  const top = sample.predictions[0];
  const topPct = top.probability * 100;
  document.getElementById("stat-top").textContent = top.cancer_type;
  document.getElementById("stat-conf").textContent = `${topPct.toFixed(1)}%`;
  document.getElementById("stat-count").textContent = sample.predictions.length;
  setGauge(topPct);
}

document.getElementById("predict-btn").addEventListener("click", async () => {
  try {
    const resp = await invoke("predict", { input: readInput() });
    renderPredictions(resp.predictions[0]);
  } catch (e) {
    setStatus(false, "Engine error");
    alert("Prediction failed: " + e);
  }
});

document.getElementById("explain-btn").addEventListener("click", async () => {
  const chart = document.getElementById("shap-chart");
  try {
    const resp = await invoke("explain", { input: readInput() });
    chart.classList.remove("chart-empty");
    chart.innerHTML = resp.svg;
  } catch (e) {
    setStatus(false, "Engine error");
    alert("Explanation failed: " + e);
  }
});
