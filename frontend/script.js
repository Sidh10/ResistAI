const ORGANISMS = [
    "Escherichia coli", "Enterobacteria spp.", "Klebsiella pneumoniae",
    "Proteus mirabilis", "Citrobacter spp.", "Morganella morganii",
    "Serratia marcescens", "Pseudomonas aeruginosa", "Acinetobacter baumannii",
    "Staphylococcus aureus", "Enterococcus spp.",
    "Streptococcus pneumoniae", "Unknown"
];

document.addEventListener("DOMContentLoaded", () => {
    const orgSelect = document.getElementById("organism");
    if (orgSelect) {
        ORGANISMS.forEach(org => {
            const opt = document.createElement("option");
            opt.value = org;
            opt.textContent = org;
            orgSelect.appendChild(opt);
        });
    }

    const predictBtn = document.getElementById("predict-btn");
    predictBtn.addEventListener("click", async () => {
        const payload = {
            organism: document.getElementById("organism").value,
            age: parseInt(document.getElementById("age").value),
            gender: document.getElementById("gender").value,
            diabetes: document.getElementById("diabetes").checked,
            hypertension: document.getElementById("hypertension").checked,
            hospital_before: document.getElementById("hospital_before").checked,
            infection_freq: parseInt(document.getElementById("infection_freq").value),
            known_ast: {}
        };

        predictBtn.textContent = "Running...";
        
        // Clear any old error
        const existingErr = document.getElementById("api-error");
        if (existingErr) existingErr.remove();

        try {
            const res = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            
            if (!res.ok) {
                throw new Error(`Server returned ${res.status}`);
            }
            const data = await res.json();
            renderResults(data);
        } catch (e) {
            console.error(e);
            const errDiv = document.createElement("div");
            errDiv.id = "api-error";
            errDiv.className = "bg-red-500/20 text-red-100 p-4 rounded-xl mt-4 border border-red-500/50 text-sm";
            errDiv.textContent = `Error connecting to API: ${e.message}. Is FastAPI running on port 8000?`;
            document.getElementById("input-form").appendChild(errDiv);
        }
        predictBtn.textContent = "Predict Resistance";
    });
});

function renderResults(data) {
    const rCount = Object.values(data.predictions).filter(p => p.label === "Resistant").length;
    const banner = document.getElementById("alert-banner");
    
    if (banner) {
        if (rCount >= 8) {
            banner.className = "mb-8 p-6 rounded-xl bg-gradient-to-r from-primary-container to-[#8e130c] flex items-center justify-between shadow-xl";
            banner.querySelector("h1").textContent = "⚠️ High Resistance Profile";
        } else if (rCount >= 4) {
            banner.className = "mb-8 p-6 rounded-xl bg-[#D4AC0D] flex items-center justify-between shadow-xl";
            banner.querySelector("h1").textContent = "⚠️ Moderate Resistance Profile";
        } else {
            banner.className = "mb-8 p-6 rounded-xl bg-[#27AE60] flex items-center justify-between shadow-xl";
            banner.querySelector("h1").textContent = "✅ Low Resistance Profile";
        }
    }

    const resCards = document.getElementById("results-cards");
    if (resCards) {
        resCards.innerHTML = '<h2 class="text-primary font-bold text-xs uppercase tracking-[0.15em] mb-4">Antibiotic Predictions</h2>';
        Object.keys(data.predictions).forEach(abx => {
            const p = data.predictions[abx];
            let colorHex = "#2E86AB"; // default
            if (p.label === "Resistant") colorHex = "#C0392B";
            else if (p.label === "Intermediate") colorHex = "#D4AC0D";
            else if (p.label === "Susceptible") colorHex = "#27AE60";
            
            const bgWidth = (p.probability * 100).toFixed(0);
            resCards.innerHTML += `
                <div class="bg-surface-container p-4 rounded-xl border-l-[4px] hover:translate-y-[-2px] transition-all duration-300 shadow-md" style="border-left-color: ${colorHex}">
                    <div class="flex justify-between items-start mb-2">
                        <span class="font-bold text-on-surface text-sm">${abx}</span>
                        <span style="color: ${colorHex}" class="font-bold">${bgWidth}%</span>
                    </div>
                    <div class="w-full bg-[#1A1A2E] h-1.5 rounded-full mb-2">
                        <div style="background-color: ${colorHex}; width: ${bgWidth}%" class="h-1.5 rounded-full"></div>
                    </div>
                    <span style="color: ${colorHex}; border: 1px solid ${colorHex}" class="text-[0.6875rem] font-bold px-2 rounded uppercase">${p.label}</span>
                </div>
            `;
        });
    }

    const shapCards = document.getElementById("shap-cards");
    if (shapCards) {
        shapCards.innerHTML = '<h2 class="text-primary font-bold text-xs uppercase tracking-[0.15em] mb-6">SHAP Feature Importance</h2><div class="relative w-full h-full min-h-[250px] flex-1"><canvas id="shap-chart"></canvas></div><p class="text-[0.6875rem] text-[#9190A5] italic mt-6 leading-relaxed">Explanation models provide insight into clinical factors driving the resistance score.</p>';
        const ctx = document.getElementById('shap-chart').getContext('2d');
        
        const labels = data.shap_summary.map(s => s.feature);
        const values = data.shap_summary.map(s => s.importance);
        const backgroundColors = values.map(v => v > 0 ? '#C0392B' : '#2E86AB');

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'SHAP Value (Impact on Model Output)',
                    data: values,
                    backgroundColor: backgroundColors,
                    borderWidth: 0,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let val = context.raw;
                                return 'Impact: ' + (val > 0 ? '+' : '') + (val*100).toFixed(1) + '%';
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: '#333348', drawBorder: false },
                        ticks: { color: '#9190A5' }
                    },
                    y: {
                        grid: { display: false, drawBorder: false },
                        ticks: { color: '#e2e0fc', font: { family: 'Inter', size: 12 } }
                    }
                }
            }
        });
    }

    const rankingCards = document.getElementById("ranking-cards");
    if (rankingCards) {
        let primaryHtml = `<h2 class="text-primary font-bold text-xs uppercase tracking-[0.15em] mb-4">Therapeutic Ranking</h2>`;
        const primary = data.primary;
        if (primary) {
            primaryHtml += `
                <div class="relative bg-surface-container-high p-6 rounded-xl border border-secondary/20 shadow-xl overflow-hidden group">
                    <span class="text-secondary font-bold text-[0.6875rem] uppercase tracking-widest mb-2 block">Primary Recommendation</span>
                    <h3 class="text-xl font-bold text-on-surface mb-2">${primary.antibiotic}</h3>
                    <div class="flex items-baseline gap-2 mb-4">
                        <span class="text-3xl font-extrabold text-[#27AE60]">${(primary.susceptibility_score*100).toFixed(0)}%</span>
                        <span class="text-[#9190A5] text-xs font-medium">Susceptible</span>
                    </div>
                </div>
            `;
        }
        
        let secondaryHtml = `<div class="bg-surface-container p-4 rounded-xl mt-4"><span class="text-[#9190A5] font-bold text-[0.6875rem] uppercase tracking-widest mb-3 block">Other Options</span><div class="space-y-3">`;
        if (data.ranked.length > 1) {
            data.ranked.slice(1, 6).forEach(r => {
                const val = (r.susceptibility_score*100).toFixed(0);
                const colorHex = r.status === "Susceptible" ? "#27AE60" : "#D4AC0D";
                secondaryHtml += `
                    <div>
                        <div class="flex justify-between items-center mb-1">
                            <span class="text-xs font-medium">${r.antibiotic}</span>
                            <span style="color: ${colorHex};" class="text-[0.6875rem] font-bold">${val}%</span>
                        </div>
                        <div class="w-full bg-[#1A1A2E] h-1.5 rounded-full">
                            <div style="background-color: ${colorHex}; width: ${val}%" class="h-1.5 rounded-full"></div>
                        </div>
                    </div>
                `;
            });
        }
        secondaryHtml += `</div></div>`;
        rankingCards.innerHTML = primaryHtml + secondaryHtml;
    }
}
