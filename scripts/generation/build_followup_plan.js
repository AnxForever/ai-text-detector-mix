const fs = require("fs");
const path = require("path");

const INPUT_PATH = path.join(__dirname, "..", "..", "new_plan_datasets", "auto_generation_plan.json");
const OUTPUT_PATH = path.join(__dirname, "..", "..", "new_plan_datasets", "auto_generation_plan_v2_followup.json");

const ATTRIBUTE_TO_TASK_TYPE = {
  "说明": "expository",
  "议论": "argumentative",
  "记叙": "narrative",
  "描写": "descriptive",
  "抒情": "lyrical"
};

const DOMAIN_RULES = [
  {
    domain: "对话",
    keywords: ["对话", "访谈", "问答", "采访", "聊天", "交流"]
  },
  {
    domain: "学术",
    keywords: ["论文", "学术", "实验", "研究", "摘要", "方法", "结论", "文献"]
  },
  {
    domain: "教育",
    keywords: ["教育", "学校", "学生", "教师", "老师", "教学", "课堂", "学习", "考试"]
  },
  {
    domain: "健康",
    keywords: ["健康", "医疗", "医学", "医院", "心理", "疾病", "诊断", "疫苗"]
  },
  {
    domain: "经济",
    keywords: ["经济", "金融", "市场", "商业", "企业", "贸易", "产业", "消费", "投资", "GDP"]
  },
  {
    domain: "文化",
    keywords: ["文化", "艺术", "文学", "历史", "传统", "哲学", "审美"]
  },
  {
    domain: "科技",
    keywords: ["科技", "人工智能", "AI", "算法", "计算", "互联网", "软件", "硬件", "机器人", "区块链", "量子", "芯片"]
  },
  {
    domain: "社会",
    keywords: ["社会", "公共", "政策", "治理", "法律", "伦理", "公平", "城市", "社区", "民生"]
  }
];

const PROMPT_TYPE_RULES = [
  { type: "dialogue", keywords: ["对话", "访谈", "问答", "采访"] },
  { type: "continuation", keywords: ["续写"] },
  { type: "rewrite", keywords: ["改写"] },
  { type: "summary", keywords: ["摘要"] }
];

const MODEL_FAMILY_WEIGHTS = [
  { name: "gpt", weight: 20 },
  { name: "claude", weight: 20 },
  { name: "gemini", weight: 20 },
  { name: "deepseek", weight: 20 },
  { name: "qwen", weight: 20 }
];

const DECODING_PROFILE_WEIGHTS = [
  { name: "low", weight: 30 },
  { name: "medium", weight: 50 },
  { name: "high", weight: 20 }
];

function pickByWeight(seed, buckets) {
  const mod = Math.abs(seed) % 100;
  let acc = 0;
  for (const bucket of buckets) {
    acc += bucket.weight;
    if (mod < acc) {
      return bucket.name;
    }
  }
  return buckets[buckets.length - 1].name;
}

function detectDomain(text) {
  for (const rule of DOMAIN_RULES) {
    if (rule.keywords.some((keyword) => text.includes(keyword))) {
      return rule.domain;
    }
  }
  return "社会";
}

function detectPromptType(text) {
  for (const rule of PROMPT_TYPE_RULES) {
    if (rule.keywords.some((keyword) => text.includes(keyword))) {
      return rule.type;
    }
  }
  return "instruction";
}

function normalizePlanItem(item) {
  const combinedText = [
    item.topic || "",
    item.genre || "",
    item.role || "",
    item.prompt || ""
  ].join(" ");

  const promptType = detectPromptType(combinedText);
  const taskType = ATTRIBUTE_TO_TASK_TYPE[item.attribute] || "other";
  const domain = detectDomain(combinedText);
  const planId = typeof item.plan_id === "number" ? item.plan_id : 0;
  const modelFamily = pickByWeight(planId + 17, MODEL_FAMILY_WEIGHTS);
  const decodingProfile = pickByWeight(planId + 59, DECODING_PROFILE_WEIGHTS);

  return {
    ...item,
    generated: false,
    generated_deepseek: false,
    generated_qwen: false,
    generated_claude: false,
    generated_parallel: false,
    domain,
    prompt_type: promptType,
    task_type: taskType,
    decoding_profile: decodingProfile,
    model_family: modelFamily,
    rewrite_type: promptType === "rewrite" ? "paraphrase" : "none",
    mix_ratio: null,
    segment_labels: [],
    source_license: "generated"
  };
}

function main() {
  if (!fs.existsSync(INPUT_PATH)) {
    console.error(`Input plan not found: ${INPUT_PATH}`);
    process.exit(1);
  }

  const raw = fs.readFileSync(INPUT_PATH, "utf8");
  const plan = JSON.parse(raw);

  if (!Array.isArray(plan)) {
    console.error("Input plan must be a JSON array.");
    process.exit(1);
  }

  const updatedPlan = plan.map((item) => normalizePlanItem(item));
  fs.writeFileSync(OUTPUT_PATH, JSON.stringify(updatedPlan, null, 2), "utf8");

  console.log(`Generated follow-up plan: ${OUTPUT_PATH}`);
  console.log(`Items: ${updatedPlan.length}`);
}

main();
