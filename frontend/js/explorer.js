let currentUid = null;

function fmtBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

async function loadExperimentList() {
  const rows = await apiGet("/api/experiments");
  const tbody = document.getElementById("exp-table-body");
  tbody.innerHTML = "";
  if (rows.length === 0) {
    document.getElementById("exp-empty").style.display = "block";
    return;
  }
  rows.forEach((r) => {
    const tr = document.createElement("tr");
    tr.className = "clickable";
    tr.innerHTML = `
      <td>
        <div>${r.name}</div>
        <div class="field-help">${r.dataset_source} \u00b7 ${r.sequence_name || "auto"}</div>
      </td>
      <td class="mono">${r.trajectory_mode}</td>
      <td><span class="badge ${r.status}">${r.status}</span></td>
    `;
    tr.addEventListener("click", () => selectExperiment(r.experiment_uid));
    tbody.appendChild(tr);
  });

  const preselect = new URLSearchParams(window.location.search).get("uid");
  if (preselect) selectExperiment(preselect);
}

async function selectExperiment(uid) {
  currentUid = uid;
  document.getElementById("no-selection").style.display = "none";
  document.getElementById("detail-panel").style.display = "block";
  document.getElementById("files-panel").style.display = "block";

  const exp = await apiGet(`/api/experiments/${uid}`);
  document.getElementById("detail-name").textContent = exp.name;
  document.getElementById("detail-meta").textContent =
    `${exp.dataset_source} \u00b7 ${exp.sequence_name || "auto"} \u00b7 mode=${exp.trajectory_mode} \u00b7 ${fmtTime(exp.created_at)}`;
  document.getElementById("config-json").textContent = JSON.stringify(exp.params_json, null, 2);
  document.getElementById("results-json").textContent = exp.results_json
    ? JSON.stringify(exp.results_json, null, 2)
    : (exp.error_message ? "ERROR:\n" + exp.error_message : "(no results yet)");
  document.getElementById("open-live-btn").href = `dashboard.html?uid=${uid}`;

  document.getElementById("folder-path").textContent = exp.output_dir;

  const filesRes = await apiGet(`/api/experiments/${uid}/files`);
  const tree = document.getElementById("file-tree");
  tree.innerHTML = "";
  if (filesRes.files.length === 0) {
    tree.innerHTML = `<div class="field-help">No files written yet.</div>`;
  }
  filesRes.files.forEach((f) => {
    const row = document.createElement("div");
    row.className = "file-row";
    row.innerHTML = `<span class="name">${f.rel_path}</span><span class="meta">${fmtBytes(f.size_bytes)}</span>`;
    row.addEventListener("click", () => previewFile(uid, f));
    tree.appendChild(row);
  });

  document.getElementById("delete-btn").onclick = async () => {
    if (!confirm(`Delete experiment "${exp.name}"? This removes it from the database (output folder is kept on disk).`)) return;
    await apiDelete(`/api/experiments/${uid}`);
    loadExperimentList();
    document.getElementById("detail-panel").style.display = "none";
    document.getElementById("files-panel").style.display = "none";
    document.getElementById("no-selection").style.display = "block";
  };
}

function previewFile(uid, f) {
  const url = `/api/experiments/${uid}/file?path=${encodeURIComponent(f.rel_path)}`;
  window.open(url, "_blank");
}

loadExperimentList();
