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

const normalize = s => s.toLowerCase().replace(/[-_]/g, ' ').replace(/[^a-z0-9\s]/g, '').split(/\s+/).filter(Boolean);

// Words that signal a food substitute. Penalised when absent from the query,
// so "bacon" doesn't match "Bacon, meatless".
const SUBSTITUTE_MARKERS = new Set([
  'meatless','imitation','substitute','artificial','flavored','flavoured',
  'analog','analogue','tofu','vegan','vegetarian',
  // game/exotic animals — penalise when not in the query so "ribeye steak"
  // doesn't match "Game meat, bison, ribeye..."
  'bison','venison','buffalo','elk','ostrich','emu','alligator','game',
  // poultry-as-substitute: "Bacon, turkey" shouldn't match "bacon";
  // safe to add because the penalty only fires when NOT in the query,
  // so "turkey bacon" / "turkey breast" etc. are unaffected.
  'turkey',
]);

// Words that transform a food into a different product category. If present
// in the description but NOT in the query, the match is likely wrong —
// e.g. "avocado" should not match "Oil, avocado".
// Also includes specific oil types (peanut, canola, soybean…) so that
// oil blends like "Oil, corn, peanut, and olive" lose to pure "Oil, olive"
// when the query is simply "olive oil".
const TYPE_TRANSFORMERS = new Set([
  'oil','juice','powder','extract','sauce','milk','cream','flour','syrup',
  'paste','spread','flakes','chips','drink','beverage','supplement',
  'sticks','jerky','nuggets','strips','burger',
  'peanut','canola','soybean','sunflower','palm','safflower','grapeseed',
]);

// Egg-part words: "egg white" or "egg yolk" are subsets of "eggs".
// If the description contains one of these AND "egg" but the query
// doesn't mention the part word, prefer a whole-egg entry instead.
const EGG_PART_WORDS = new Set(['white', 'yolk', 'albumen']);

// Check whether a USDA food search-result entry has any useful macro data.
// Entries with zero calories + zero protein + zero fat have missing data in
// the USDA database and should be strongly de-ranked.
function hasUsefulNutrientData(food) {
  if (!food.foodNutrients?.length) return false;
  const sum = food.foodNutrients.reduce((acc, n) => {
    if ([1008, 2047, 2048, 1003, 1004].includes(n.nutrientId)) acc += (n.value ?? 0);
    return acc;
  }, 0);
  return sum > 0;
}

// Score how well a USDA food description matches the query.
// Returns 0–1; higher is better.
function matchScore(query, description, { curatedBonus = 0, hasData = true } = {}) {
  const queryWords = normalize(query);
  const descWordArr = normalize(description);
  const descWords   = new Set(descWordArr);

  if (queryWords.length === 0) return 0;

  // Stem-aware hit: a query word matches if it equals a desc word OR if
  // one is a prefix of the other (min 5 chars), catching plural/singular
  // mismatches like "avocado" vs "avocados".
  function hits(qw) {
    if (descWords.has(qw)) return true;
    if (qw.length >= 5) {
      for (const dw of descWords) {
        if (dw.startsWith(qw) || qw.startsWith(dw)) return true;
      }
    }
    return false;
  }

  const matchedCount = queryWords.filter(hits).length;
  const recall = matchedCount / queryWords.length;

  // Penalise long descriptions (branded items tend to be verbose)
  const lengthPenalty = Math.min(1, 10 / Math.max(10, descWordArr.length));

  // Core-noun guard: the last query word is the primary food (e.g. "bacon").
  // If absent from description, crush the score so modifier-only matches lose.
  const coreWord = queryWords[queryWords.length - 1];
  const coreBoost = hits(coreWord) ? 1.0 : 0.15;

  // Substitute penalty: description contains a substitute marker the query
  // doesn't mention → likely wrong product.
  const hasUnwantedSubstitute = [...SUBSTITUTE_MARKERS].some(
    m => descWords.has(m) && !queryWords.includes(m)
  );

  // Transformer penalty: description contains a category-shifting word
  // (oil, juice, powder…) the query doesn't mention → wrong form of the food.
  const hasUnwantedTransformer = [...TYPE_TRANSFORMERS].some(
    m => descWords.has(m) && !queryWords.includes(m)
  );

  // Egg-part penalty: "egg white" / "egg yolk" are subsets of "eggs".
  // If the description has an egg-part word and the query doesn't, prefer whole.
  const isEggDescription = descWords.has('egg') || descWords.has('eggs');
  const hasUnwantedEggPart = isEggDescription && [...EGG_PART_WORDS].some(
    m => descWords.has(m) && !queryWords.includes(m)
  );

  // Missing-data penalty: USDA entries with zero calories + protein + fat are
  // placeholder rows with no nutritional information — rank them near the bottom.
  const dataPenalty = hasData ? 1.0 : 0.1;

  const penalty = hasUnwantedSubstitute ? 0.2
    : hasUnwantedTransformer ? 0.5
    : hasUnwantedEggPart     ? 0.6
    : 1.0;

  return (recall * 0.8 + lengthPenalty * 0.2) * coreBoost * penalty * dataPenalty + curatedBonus;
}

function bestMatch(query, foods) {
  let best = null, bestScore = -1;
  for (const food of foods) {
    const curatedBonus = food.dataType === 'Foundation' ? 0.15
      : food.dataType === 'SR Legacy' ? 0.05
      : 0;
    const score = matchScore(query, food.description, {
      curatedBonus,
      hasData: hasUsefulNutrientData(food),
    });
    if (score > bestScore) { bestScore = score; best = food; }
  }
  return best;
}

async function fetchUSDA(query, dataType) {
  // URLSearchParams encodes commas and spaces in dataType, which USDA rejects.
  // Build base params normally then append dataType with only spaces encoded.
  const params = new URLSearchParams({ query, pageSize: '8', api_key: USDA_API_KEY });
  const url = `${USDA_BASE}/foods/search?${params}&dataType=${encodeURIComponent(dataType).replace(/%2C/g, ',')}`;
  const res = await fetch(url);
  if (!res.ok) return [];
  const data = await res.json();
  return data.foods ?? [];
}

async function searchUSDA(foodName) {
  // Fetch curated and branded pools in parallel. Merging them before scoring
  // ensures Foundation/SR Legacy entries are always in the candidate pool —
  // a single combined API call lets USDA's own ranking bury them past our
  // page limit when a branded exact-name match sits at position 1.
  const [curated, branded] = await Promise.all([
    fetchUSDA(foodName, 'Foundation,SR Legacy'),
    fetchUSDA(foodName, 'Branded,Survey (FNDDS)'),
  ]);
  const all = [...curated, ...branded];
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
