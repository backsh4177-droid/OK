import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";

env.allowLocalModels = false;

const statusEl = document.getElementById("status");
const runBtn = document.getElementById("runBtn");
const swapBtn = document.getElementById("swapBtn");
const scoreEl = document.getElementById("score");
const interpretationEl = document.getElementById("interpretation");
const meterFill = document.getElementById("meterFill");
const resultCard = document.getElementById("resultCard");

const textAEl = document.getElementById("textA");
const textBEl = document.getElementById("textB");

const keywordEl = document.getElementById("keyword");
const fillBtn = document.getElementById("fillBtn");
const triBtn = document.getElementById("triBtn");
const neutralEl = document.getElementById("neutral");
const positiveEl = document.getElementById("positive");
const negativeEl = document.getElementById("negative");
const triTable = document.getElementById("triTable");

let extractor;

function cosineSimilarity(a, b) {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function similarityLabel(score) {
  if (score >= 0.8) return "매우 유사한 의미로 해석될 가능성이 큽니다.";
  if (score >= 0.6) return "비슷한 맥락으로 묶일 가능성이 높습니다.";
  if (score >= 0.4) return "일부 맥락만 공유합니다.";
  if (score >= 0.2) return "연관은 있지만 의미적 거리가 있습니다.";
  return "거의 다른 의미 영역으로 보입니다.";
}

async function getEmbedding(text) {
  const output = await extractor(text, { pooling: "mean", normalize: true });
  return Array.from(output.data);
}

async function ensureModel() {
  if (extractor) return;
  statusEl.textContent = "모델을 불러오는 중... (처음 1회는 10~30초 소요)";
  runBtn.disabled = true;
  triBtn.disabled = true;

  extractor = await pipeline("feature-extraction", "Xenova/paraphrase-multilingual-MiniLM-L12-v2");

  statusEl.textContent = "준비 완료. 문장을 입력하고 계산하세요.";
  runBtn.disabled = false;
  triBtn.disabled = false;
}

function fillExampleSentences() {
  const k = keywordEl.value.trim() || "차갑다";
  neutralEl.value = `${k}라는 표현은 맥락에 따라 감정과 상태를 함께 설명한다.`;
  positiveEl.value = `그 사람은 판단이 ${k}서 감정에 휘둘리지 않고 침착하게 결정을 내린다.`;
  negativeEl.value = `요즘 그는 말투가 너무 ${k}서 주변 사람들이 상처를 받는다.`;
}

async function runPair() {
  const a = textAEl.value.trim();
  const b = textBEl.value.trim();
  if (!a || !b) {
    statusEl.textContent = "입력 A/B를 모두 채워주세요.";
    return;
  }

  try {
    await ensureModel();
    statusEl.textContent = "임베딩 계산 중...";
    const [va, vb] = await Promise.all([getEmbedding(a), getEmbedding(b)]);
    const score = cosineSimilarity(va, vb);

    scoreEl.textContent = score.toFixed(4);
    interpretationEl.textContent = similarityLabel(score);
    meterFill.style.width = `${Math.max(0, Math.min(1, (score + 1) / 2)) * 100}%`;
    resultCard.hidden = false;
    statusEl.textContent = "완료.";
  } catch (error) {
    statusEl.textContent = `오류: ${error.message}`;
  }
}

async function runTriad() {
  const samples = [neutralEl.value.trim(), positiveEl.value.trim(), negativeEl.value.trim()];
  if (samples.some((x) => !x)) {
    statusEl.textContent = "중립/긍정/부정 문장을 모두 채워주세요.";
    return;
  }

  try {
    await ensureModel();
    statusEl.textContent = "3문장 비교 계산 중...";

    const [n, p, ng] = await Promise.all(samples.map((t) => getEmbedding(t)));
    const rows = [
      ["중립 ↔ 긍정", cosineSimilarity(n, p)],
      ["중립 ↔ 부정", cosineSimilarity(n, ng)],
      ["긍정 ↔ 부정", cosineSimilarity(p, ng)],
    ];

    const tbody = triTable.querySelector("tbody");
    tbody.innerHTML = rows
      .map(([label, s]) => `<tr><td>${label}</td><td>${s.toFixed(4)}</td></tr>`)
      .join("");

    triTable.hidden = false;
    statusEl.textContent = "3문장 비교 완료.";
  } catch (error) {
    statusEl.textContent = `오류: ${error.message}`;
  }
}

runBtn.addEventListener("click", runPair);
swapBtn.addEventListener("click", () => {
  const temp = textAEl.value;
  textAEl.value = textBEl.value;
  textBEl.value = temp;
});

fillBtn.addEventListener("click", fillExampleSentences);
triBtn.addEventListener("click", runTriad);

fillExampleSentences();
ensureModel();
