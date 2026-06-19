import 'dotenv/config';
import express from 'express';
import Anthropic from '@anthropic-ai/sdk';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const app = express();
app.use(express.json());
app.use(express.static(join(__dirname, 'public')));

const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
const USDA_API_KEY = process.env.USDA_API_KEY || 'DEMO_KEY';
const USDA_BASE = 'https://api.nal.usda.gov/fdc/v1';

async function parseMealWithClaude(description) {
  const message = await anthropic.messages.create({
    model: 'claude-haiku-4-5-20251001',
    max_tokens: 1024,
    messages: [{
      role: 'user',
      content: `Parse this meal description into a JSON array of food items with estimated weights in grams.
Convert casual units to grams using these approximations:
- 1 cup (rice/pasta cooked) ≈ 200g, (leafy greens) ≈ 30g, (liquid) ≈ 240g, (nuts) ≈ 120g
- 1 handful ≈ 30g
- 1 tablespoon ≈ 15g
- 1 teaspoon ≈ 5g
- 1 slice bread ≈ 30g, cheese ≈ 25g, deli meat ≈ 28g
- 1 medium apple/banana/orange ≈ 150g
- 1 egg ≈ 50g
- 1 piece chicken breast ≈ 170g

Return ONLY valid JSON array, no explanation:
[{"food_name": "...", "grams": 123}, ...]

Meal: ${description}`
    }]
  });

  const text = message.content[0].text.trim();
  const match = text.match(/\[[\s\S]*\]/);
  if (!match) throw new Error('Claude did not return valid JSON');
  return JSON.parse(match[0]);
}

// Score how well a USDA food description matches the query.
// Returns 0–1; higher is better.
function matchScore(query, description) {
  const normalize = s => s.toLowerCase().replace(/[-_]/g, ' ').replace(/[^a-z0-9\s]/g, '').split(/\s+/).filter(Boolean);
  const queryWords = normalize(query);
  const descWords  = new Set(normalize(description));

  if (queryWords.length === 0) return 0;

  // Fraction of query words found in the description
  const hits = queryWords.filter(w => descWords.has(w)).length;
  const recall = hits / queryWords.length;

  // Penalise very long descriptions (branded items tend to be verbose)
  const lengthPenalty = Math.min(1, 10 / Math.max(10, descWords.size));

  // The last word in the query is the primary food noun (e.g. "bacon" in
  // "low-fat bacon"). If it's absent from the description, heavily penalise
  // so that modifier-only matches (e.g. "Buttermilk, low fat") don't win.
  const coreWord = queryWords[queryWords.length - 1];
  const coreBoost = descWords.has(coreWord) ? 1.0 : 0.15;

  return (recall * 0.85 + lengthPenalty * 0.15) * coreBoost;
}

function bestMatch(query, foods) {
  let best = null, bestScore = -1;
  for (const food of foods) {
    const score = matchScore(query, food.description);
    if (score > bestScore) { bestScore = score; best = food; }
  }
  return best;
}

async function fetchUSDA(query, dataType) {
  const params = new URLSearchParams({
    query,
    pageSize: '5',
    dataType,
    api_key: USDA_API_KEY,
  });
  const res = await fetch(`${USDA_BASE}/foods/search?${params}`);
  if (!res.ok) throw new Error(`USDA search failed: ${res.status}`);
  const data = await res.json();
  return data.foods ?? [];
}

async function searchUSDA(foodName) {
  // Prefer Foundation + SR Legacy (curated, generic foods)
  const curated = await fetchUSDA(foodName, 'Foundation,SR Legacy');
  if (curated.length > 0) return bestMatch(foodName, curated);

  // Fall back to all data types (includes Branded) if nothing curated found
  const all = await fetchUSDA(foodName, 'Foundation,SR Legacy,Branded,Survey (FNDDS)');
  if (all.length === 0) return null;
  return bestMatch(foodName, all);
}

function getNutrient(food, nutrientIds) {
  if (!food.foodNutrients) return 0;
  for (const id of nutrientIds) {
    const n = food.foodNutrients.find(n => n.nutrientId === id);
    if (n && n.value != null) return n.value;
  }
  return 0;
}

// Nutrient IDs from USDA FoodData Central
const NUTRIENT_IDS = {
  calories: [1008, 2047, 2048],
  protein:  [1003],
  carbs:    [1005],
  fat:      [1004],
};

app.post('/api/analyze', async (req, res) => {
  try {
    const { description } = req.body;
    if (!description?.trim()) return res.status(400).json({ error: 'No meal description provided' });

    const items = await parseMealWithClaude(description);

    const results = await Promise.all(items.map(async (item) => {
      try {
        const food = await searchUSDA(item.food_name);
        if (!food) {
          return { ...item, found: false, calories: 0, protein: 0, carbs: 0, fat: 0 };
        }

        const per100 = {
          calories: getNutrient(food, NUTRIENT_IDS.calories),
          protein:  getNutrient(food, NUTRIENT_IDS.protein),
          carbs:    getNutrient(food, NUTRIENT_IDS.carbs),
          fat:      getNutrient(food, NUTRIENT_IDS.fat),
        };

        const scale = item.grams / 100;
        return {
          food_name: item.food_name,
          matched_name: food.description,
          grams: item.grams,
          found: true,
          calories: Math.round(per100.calories * scale),
          protein:  Math.round(per100.protein  * scale * 10) / 10,
          carbs:    Math.round(per100.carbs    * scale * 10) / 10,
          fat:      Math.round(per100.fat      * scale * 10) / 10,
        };
      } catch {
        return { ...item, found: false, calories: 0, protein: 0, carbs: 0, fat: 0 };
      }
    }));

    const totals = results.reduce((acc, r) => ({
      calories: acc.calories + r.calories,
      protein:  Math.round((acc.protein  + r.protein)  * 10) / 10,
      carbs:    Math.round((acc.carbs    + r.carbs)    * 10) / 10,
      fat:      Math.round((acc.fat      + r.fat)      * 10) / 10,
    }), { calories: 0, protein: 0, carbs: 0, fat: 0 });

    res.json({ items: results, totals });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));
