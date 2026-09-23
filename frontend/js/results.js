let currentReportId = null;

function fmtPct(x) {
  return x === null || x === undefined ? "n/a" : `${(x * 100).toFixed(1)}%`;
}

function renderSummary(summary) {
  currentReportId = summary.report_id;
  document.getElementById("report-content").style.display = "block";

  document.getElementById("stat-raw-rate").textContent =
    `${fmtPct(summary.raw_recovery_rate)}`;
  document.getElementById("stat-hom-rate").textContent =
    `${fmtPct(summary.homography_recovery_rate)}`;
  document.getElementById("stat-pvalue").textContent =
    summary.mcnemar_p_value !== null && summary.mcnemar_p_value !== undefined
      ? summary.mcnemar_p_value.toExponential(2) : "n/a";
  document.getElementById("stat-or").textContent =
    summary.odds_ratio
      ? `${summary.odds_ratio.toFixed(2)} [${summary.odds_ratio_ci_low?.toFixed(2)}, ${summary.odds_ratio_ci_high?.toFixed(2)}]`
      : "n/a";
  document.getElementById("stat-n-gaps").textContent = summary.n_gaps_total ?? "0";
  document.getElementById("stat-n-seq").textContent = summary.n_sequences ?? "0";
  document.getElementById("stat-n-cat").textContent = summary.n_categories ?? "0";
  document.getElementById("stat-n-runs").textContent = summary.n_runs ?? "0";
  document.getElementById("report-meta").textContent =
    `Report ${summary.report_id} \u00b7 prefix="${summary.name_prefix || "(all)"}" \u00b7 generated ${fmtTime(summary.generated_at)} \u00b7 raw ${summary.raw_recovered}/${summary.raw_total} vs homography ${summary.homography_recovered}/${summary.homography_total} \u00b7 ${summary.n_discordant_pairs} discordant pairs (${summary.favor_homography} favor homography, ${summary.favor_raw} favor raw)`;

  const figGrid = document.getElementById("fig-grid");
  figGrid.innerHTML = "";
  (summary.figures || []).forEach((fig) => {
    if (fig.error) return;
    const pngUrl = `/api/reports/${summary.report_id}/file?path=figures/${fig.name}.png`;
    const pdfUrl = `/api/reports/${summary.report_id}/file?path=figures/${fig.name}.pdf`;
    const card = document.createElement("div");
    card.className = "fig-card";
    card.innerHTML = `
      <a href="${pngUrl}" target="_blank"><img src="${pngUrl}" alt="${fig.title}" loading="lazy" /></a>
      <div class="fig-card-body">
        <span class="fig-card-title">${fig.title}</span>
        <a href="${pdfUrl}" target="_blank">PDF &darr;</a>
      </div>`;
    figGrid.appendChild(card);
  });

  const tablesContainer = document.getElementById("tables-container");
  tablesContainer.innerHTML = "";
  (summary.tables || []).forEach(async (tbl) => {
    const wrap = document.createElement("div");
    wrap.style.marginBottom = "22px";
    wrap.innerHTML = `<h2 style="font-size:13px; color:var(--text-dim); margin-bottom:8px;">${tbl.title}
      <a href="/api/reports/${summary.report_id}/file?path=tables/${tbl.name}.csv" target="_blank"
         style="font-family:var(--font-mono); font-size:11px; margin-left:10px;">CSV &darr;</a></h2>
      <div style="overflow-x:auto;"><table class="data-table" id="tbl-${tbl.name}"></table></div>`;
    tablesContainer.appendChild(wrap);

    try {
      const rows = await apiGet(`/api/reports/${summary.report_id}/file?path=tables/${tbl.name}.json`);
      const tableEl = document.getElementById(`tbl-${tbl.name}`);
      if (!rows.length) { tableEl.innerHTML = "<tr><td>(empty)</td></tr>"; return; }
      const cols = Object.keys(rows[0]);
      let html = "<thead><tr>" + cols.map(c => `<th>${c}</th>`).join("") + "</tr></thead><tbody>";
      rows.forEach(r => {
        html += "<tr>" + cols.map(c => `<td>${r[c] === null ? "&mdash;" : r[c]}</td>`).join("") + "</tr>";
      });
      html += "</tbody>";
      tableEl.innerHTML = html;
    } catch (e) {
      document.getElementById(`tbl-${tbl.name}`).innerHTML = `<tr><td>Failed to load: ${e.message}</td></tr>`;
    }
  });
}

async function loadPastReports() {
  const listEl = document.getElementById("past-reports-list");
  try {
    const reports = await apiGet("/api/reports");
    if (!reports.length) {
      listEl.innerHTML = "No reports generated yet.";
      return;
    }
    listEl.innerHTML = "";
    const table = document.createElement("table");
    table.innerHTML = "<thead><tr><th>Generated</th><th>Prefix</th><th>Runs</th><th>Gaps</th><th>p-value</th></tr></thead>";
    const tbody = document.createElement("tbody");
    reports.forEach((r) => {
      const tr = document.createElement("tr");
      tr.className = "clickable";
      tr.innerHTML = `<td>${fmtTime(r.generated_at)}</td><td class="mono">${r.name_prefix || "(all)"}</td>
        <td>${r.n_runs}</td><td>${r.n_gaps_total}</td>
        <td>${r.mcnemar_p_value !== null && r.mcnemar_p_value !== undefined ? r.mcnemar_p_value.toExponential(2) : "n/a"}</td>`;
      tr.addEventListener("click", () => renderSummary(r));
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    listEl.appendChild(table);
  } catch (e) {
    listEl.innerHTML = `Failed to load past reports: ${e.message}`;
  }
}

document.getElementById("generate-btn").addEventListener("click", async () => {
  const btn = document.getElementById("generate-btn");
  const statusEl = document.getElementById("generate-status");
  const prefix = document.getElementById("prefix-input").value.trim();
  btn.disabled = true;
  statusEl.textContent = "Generating (crunching all matching experiments, may take a few seconds)...";
  try {
    const summary = await apiPost("/api/reports/generate", { name_prefix: prefix });
    statusEl.textContent = "Done.";
    renderSummary(summary);
    loadPastReports();
  } catch (e) {
    statusEl.textContent = "Failed: " + e.message;
  } finally {
    btn.disabled = false;
  }
});

loadPastReports();

// Auto-load the latest report on page open, if one exists
(async () => {
  try {
    const latest = await apiGet("/api/reports/latest");
    renderSummary(latest);
  } catch (e) {
    // no reports yet -- fine, the generate button is right there
  }
})();
