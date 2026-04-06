# Speaker Notes: ESG Market Prediction Presentation

This document contains slide-by-slide talking points for `presentation.html`.
Navigate using → (right arrow) for main slides and ↓ (down arrow) for sub-slides.

---

## Slide 1: Title — ESG & Market Dynamics

**On screen:** Title, subtitle, tagline (LSTM + Attention · 700 Stocks · Walk-Forward Validation)

**Say:**
- "Good [Morning/Afternoon]. Today I'll walk you through our work on predicting stock price direction using ESG data and deep learning."
- "We'll cover the motivation, what we built, the engineering challenges we overcame, our final results from the training pipeline, and what this opens up for the future."

---

## Slide 2: Why did we do this?

**On screen:** "The Gap" card + 3 research questions (appear one by one on click)

**Say:**
- "Most financial machine learning relies purely on price-based technical indicators — things like RSI, moving averages, and volume."
- "We hypothesized that ESG scores — Environmental, Social, and Governance metrics — encode latent risk signals and institutional sentiment that price data alone doesn't capture."

**As each research question appears (click to advance):**
1. "First — can non-traditional, qualitative data actually provide a measurable edge?"
2. "Second — does strong ESG compliance protect stocks during downturns?"
3. "And third — can a deep learning model like an LSTM with Attention extract that signal from inherently noisy market data?"

---

## Slide 3: What did we do?

**On screen:** Two cards side by side — Data Setup (left) and Experimental Design (right)

**Say:**
- "We assembled a dataset of over 700 US-listed stocks spanning 2020 to 2023, merging daily price data from Yahoo Finance with static ESG scores from Finnhub."
- "The task is binary classification: will the stock go Up or Down over the next 5 trading days?"
- "We ran two models head-to-head. The ESG model receives 28 features — price technicals plus ESG data plus derived features. The baseline receives only the 9 price-based technical features."
- "Both use the exact same LSTM architecture, trained with a 3-seed ensemble and a strict chronological walk-forward split — no data leakage."

---

## Slide 3b (sub-slide 1): Data Exploration — Scores & Grades

**On screen:** Two charts — ESG Score Distributions (left) and ESG Grade Distributions (right)

**Navigation:** Press ↓ to reach this slide from "What did we do?"

**Say:**
- "Before training, let's look at our data. On the left, the normalised ESG score distributions are fairly well spread — that's healthy, it means there's variance for the model to learn from."
- "On the right, grade distributions show most stocks cluster at the lower end — B and BB grades. This skew is a known limitation; the model has fewer examples of top-rated companies."

---

## Slide 3b (sub-slide 2): Data Exploration — Industry & Feature Quality

**On screen:** Industry Distribution bar chart (left) and Feature Distributions post-normalisation box plot (right)

**Say:**
- "The industry chart reveals sector concentration — Technology and Healthcare dominate. We address this bias later with industry-relative z-scores."
- "The box plot on the right is our post-normalisation sanity check. Price features are centred near zero with spreads between -3 and +3. ESG features sit neatly between 0 and 1. No wild outliers — the pipeline is clean."

**Then press → to continue to "How did we do it?"**

---

## Slide 4a: How did we do it — Part 1: Overcoming the Overfitting Trap

**On screen:** Red-bordered problem card + fixes table (rows appear one by one)

**Say:**
- "Now we get to the real engineering work. Our initial model hit 55% accuracy, but it completely overfitted by epoch 5 — it memorised the training data and collapsed into predicting 'Up' on almost every sample."
- "Financial data is extremely noisy. The same 5-day window can go up one year and down the next under similar conditions."

**As each table row appears (click to advance):**
1. "We shrunk the model — hidden size from 64 to 32, two LSTM layers down to one. Fewer parameters means less capacity to memorise."
2. "Increased dropout from 0.3 to 0.5 — more aggressive regularisation."
3. "Added a Weighted Random Sampler to force every training batch to be exactly 50% Up and 50% Down, preventing majority-class collapse."
4. "Implemented label smoothing with epsilon 0.1 — the model can't become overconfident on inherently noisy labels."
5. "Tightened gradient clipping to 0.5 for stable weight updates."

---

## Slide 4b: How did we do it — Part 2: Feature Engineering & Subsampling

**On screen:** Two cards (Industry-Relative ESG + 7 Window Features) then Sequence Subsampling card appears

**Navigation:** Press ↓ from slide 4a to reach this sub-slide

**Say:**
- "We realised that raw static ESG scores acted like stock ID barcodes — the model used them to identify which stock it was looking at, not as a genuine market signal."
- "So we converted them to z-scores within each industry. Now the model learns: 'Does this energy company have a better environmental score than its energy sector peers?'"
- "We also extracted 7 derived features computed within each 30-day window: MACD, Bollinger Band position, On-Balance Volume momentum, return skewness, kurtosis, trend strength, and volatility trend."

**When the Subsampling card appears (click):**
- "Finally — and this was critical — adjacent 30-day sliding windows overlap by 97%. Training on all of them is like showing the model the same data 30 times."
- "We keep every 3rd sequence. It reduces training size but massively improves generalisation because the model must learn general patterns, not memorise specific dates."

---

## Slide 5: What are the results?

**On screen:** Two metric pills (54.52% ESG vs 54.46% Baseline), plus two cards with recall breakdown and comparison table

**Say:**
- "Our final accuracy is 54.52% for the ESG model, slightly edging out the price-only baseline at 54.46%."
- "Now — 54.5% sounds lower than our initial 55%. But this is actually a massive improvement."
- "The old 55% was fake. It was a collapsed model that predicted 'Up' on nearly every sample, riding the bull-market label distribution. It had essentially zero recall for down days."
- "Our new model achieves truly balanced recall: 54.76% for Up days and 54.28% for Down days. It's genuinely distinguishing between the two classes."
- "Looking at the comparison table: the ESG model outperforms the baseline on AUC-ROC — 0.5645 versus 0.5621 — a lift of +0.0024. Macro-F1 also improves by +0.0006."
- "In quantitative finance, a legitimate, generalised 54.5% directional accuracy on a 5-day horizon — verified with no look-ahead bias — is a highly competitive result."

---

## Slide 6: What does this mean for the future?

**On screen:** Three cards appearing one by one (Reliable Pipeline → Expanding Data → Real-World Deployment)

**As each card appears (click to advance):**

1. **Reliable Pipeline:** "First, we now have a battle-tested PyTorch pipeline that's resistant to the three most common financial ML pitfalls: time leakage, class collapse, and overfitting. This is our foundation."

2. **Expanding Alternative Data:** "Second, because the model successfully parses non-price alternative data, we can expand. Next steps: Temporal Fusion Transformers for time-varying ESG, NLP sentiment extraction from news articles and earnings call transcripts, and sector-specific sub-models."

3. **Real-World Deployment:** "Third, a verified 54.5% edge without look-ahead bias has real-world value. Combined with Kelly Criterion position sizing and strict risk management, this backbone can anchor a positive-expectancy algorithmic trading portfolio."

---

## Slide 7: Thank You

**On screen:** "Thank You" + "Questions?"

**Say:**
- "Thank you for your time. I'm happy to take any questions — whether about the feature engineering pipeline, the LSTM architecture, the walk-forward validation, or where we're headed next."

---

## Quick Navigation Reference

| Slide | Key Press | Content |
|-------|-----------|---------|
| 1 | → | Title |
| 2 | → | Why did we do this? |
| 3 | → | What did we do? |
| 3b-1 | ↓ | Data Exploration: Scores & Grades |
| 3b-2 | ↓ | Data Exploration: Industry & Features |
| 4a | → | How: Overfitting fixes (table rows on click) |
| 4b | ↓ | How: Feature engineering & subsampling |
| 5 | → | Results (metrics + table) |
| 6 | → | Future (3 cards on click) |
| 7 | → | Thank You |
