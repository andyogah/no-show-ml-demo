# Intro slides: the first ten minutes

Twelve slides, ten minutes, ending on the handoff into the notebook. Each slide lists what is
on screen, what to say, and the clock. Figures referenced here are in `docs/figures/`, built
by `tools/make_intro_figures.py` from the same synthetic data as the notebook, so every number
on a slide matches what the room sees later.

The spine of the whole talk is one line: **y = mx + b**. The intro draws it, the notebook
finds it, the governance section asks who owns it.

| # | Slide | Min | Cumulative |
|---|---|---|---|
| 1 | Title | 0.5 | 0.5 |
| 2 | The word cloud | 0.5 | 1 |
| 3 | One idea under all of them | 0.5 | 1.5 |
| 4 | Definitions, nested | 3 | 4.5 |
| 5 | What the data says (dots) | 0.5 | 5 |
| 6 | y = mx + b (the line) | 1 | 6 |
| 7 | Learning is finding m and b | 0.5 | 6.5 |
| 8 | More inputs, more m's | 0.5 | 7 |
| 9 | Squash it | 0.5 | 7.5 |
| 10 | Stack it | 1 | 8.5 |
| 11 | The problem for the next 32 minutes | 1 | 9.5 |
| 12 | Handoff | 0.5 | 10 |

Running long at slide 4: skip straight to slide 5. The definitions are where every AI talk
bleeds time.

---

## 1. Title

**On screen:** "Developing AI/ML. What it actually takes." Your name, role, date.

**Say:** One sentence on who you are and one on what the hour is: "I build and review this
stuff for a living. In the next hour you will watch a model get built, and then I will show
you why that was the easy part."

## 2. The word cloud

**On screen:** Forty-odd terms squished together, no hierarchy, mixed sizes. Artificial
intelligence, machine learning, deep learning, neural network, data science, statistics,
regression, classification, large language model, GPT, transformer, embedding, RAG, vector
database, prompt engineering, fine-tuning, agent, agentic AI, multi-agent, MCP, copilot,
generative AI, predictive analytics, computer vision, NLP, reinforcement learning, supervised,
unsupervised, feature, label, training, inference, overfitting, bias, drift, MLOps,
responsible AI, hallucination, guardrails, tokens, GPU.

**Say:** Nothing for three seconds. Then: "Which of these has someone asked you about this
month?" Let two or three people answer. Do not read the slide.

## 3. One idea under all of them

**On screen:** The same cloud, faded to grey, with one line in the middle: **y = mx + b**.

**Say:** "Nearly every word on that slide sits on one idea, and it is the one you learned at
fourteen. Hold that thought for two minutes while we get the definitions out of the way."

## 4. Definitions, nested

**On screen:** Five concentric rings, outer to inner: AI, machine learning, deep learning.
Beside the rings, two boxes stacked: agent, agentic AI, with an arrow from the rings into
them ("a model, plus"). One sentence next to each term, all phrased on the no-show problem.

| Term | The sentence on the slide |
|---|---|
| Artificial intelligence | The computer does something that looks like judgement, such as deciding which appointments are at risk. |
| Machine learning | It learned the rule from past appointments instead of a person writing the rule. |
| Deep learning | The same thing with many stacked layers. Needed for images, text, speech and signals. |
| Agent | A model that can also take actions, like sending the reminder, and check what happened. |
| Agentic AI | Agents chained together toward a goal, choosing their own next steps. |

**Say:** Walk the rings from the outside in, one sentence each, no elaboration. Then the two
boxes. Then: "Statistics is the one not on this slide. It is the discipline of knowing how
wrong you are, and it shows up the moment we evaluate a model." Hard stop at three minutes.
If someone wants to argue a definition, offer to take it in Q&A.

## 5. What the data says

**On screen:** `figures/01-line.png`. Dots only: no-show rate against days between scheduling
and the appointment.

**Say:** "Here is our outpatient scheduling data. Every dot is a group of appointments. The
further out you book, the more people do not turn up. Everyone in this building knows that
already. The question is what you do with it."

## 6. y = mx + b

**On screen:** `figures/02-line-fitted.png`. The line through the dots, with b and m called
out.

**Say:** "Draw a line through it. b is where the line starts, the no-show rate for someone
booked today, about four percent. m is the slope: each extra ten days of waiting costs about
seven points. That is the whole model. Give me a lead time, I give you a no-show rate."

## 7. Learning is finding m and b

**On screen:** The same figure, with the words **machine learning = finding m and b from the
data instead of guessing them** underneath.

**Say:** "Nobody typed four percent and seven points. The computer found them by trying
values, measuring how wrong the line was, and nudging. That nudging loop is what the word
training means. You will watch it run in about twenty minutes."

## 8. More inputs, more m's

**On screen:** y = m₁·lead time + m₂·prior no-shows + m₃·reminder sent + m₄·distance + ... + b.
Then the same thing with the m's relabelled **weights**.

**Say:** "Lead time is not the only thing that matters. Prior no-shows, whether a reminder went
out, how far they live. Each one gets its own m. The m's are what people call weights. A
model with a billion parameters is this equation with a billion m's."

## 9. Squash it

**On screen:** `figures/03-squashed.png`. The straight line sailing past 100%, and the same
idea bent so it levels off.

**Say:** "A straight line has one problem. Book far enough out and it predicts a hundred and
ten percent of people no-show. So we bend it to stay between zero and one hundred. That bent
line is logistic regression, and it is the first model we train today."

## 10. Stack it

**On screen:** Three rows, building down:
1. one squashed line (logistic regression)
2. several squashed lines feeding into another squashed line (a neural network)
3. many of those stacked (deep learning), with the words "images, text, speech, signals"

**Say:** "Stack a few of those bent lines so the output of one feeds the next, and you have a
neural network. Stack a lot, and you have deep learning. The large language models are this,
with a trick for paying attention to words in order, run on an enormous amount of text. It is
the same equation all the way down. Every one of those weights was found by the same
nudging loop."

## 11. The problem for the next 32 minutes

**On screen:** Four lines.
- Predict which outpatient appointments will be no-shows.
- The day before, for the scheduling team.
- Synthetic data, generated in front of you. No PHI anywhere.
- Watch for three traps I planted.

**Say:** "Same problem, real code. The data is made up, on purpose, so you can take this home
and run it. I have hidden three traps in it, the kind we find in real projects, and I will
show you each one when the model falls into it."

## 12. Handoff

**On screen:** "Part 2: build it. 32 minutes." A screenshot of the notebook's first cell.

**Say:** "Here is the loop finding m and b." Switch to the notebook, block 1.

---

## After the notebook

The notebook ends on "who is watching this number?" That is the opening line of the
governance section. Suggested spine for its 15 minutes, one slide each:

1. That took 32 minutes. Here is why it takes six months.
2. Data: access, ownership, what "no-show" means in three different systems.
3. The bar: what accuracy means clinically, and who agrees to it.
4. The leak: someone has to know how every column is populated.
5. The group the model missed: fairness review and what you do about it.
6. The threshold: a decision the clinic owns, written down, revisited.
7. The drift: monitoring, retraining triggers, someone on call.
8. Models versus agents: the moment the thing can act, the review changes.
9. The intake and risk review path at our organisation.
10. What to bring me first.

Then the two seeded questions from `talk-track.md`.
