# Intro script: the first ten minutes

The intro runs **inside the notebook**, as Part 1 at the top, so there is no switching between
a deck and JupyterLab. This file is the script for those cells: what is on screen, what to say,
and the clock. "Slide" below means one scroll-stop in the notebook. Figures are in
`docs/figures/`, built by `tools/make_intro_figures.py` from the same synthetic data as the
notebook, so every number on screen matches what the room sees later. The image cells have
their inputs collapsed; if JupyterLab shows the code, click the blue bar to the left of the
cell.

If you decide to use a deck after all, the same sequence maps one slide per section below.

The spine of the whole talk is one line: **y = mx + b**. The intro draws it, the notebook
finds it, the governance section asks who owns it.

| # | Slide | Min | Cumulative |
|---|---|---|---|
| 1 | Title | 0.5 | 0.5 |
| 2 | The word cloud | 0.5 | 1 |
| 3 | One idea under all of them | 0.5 | 1.5 |
| 4 | Six colours, six ideas | 2.5 | 4 |
| 4b | One problem for the whole hour | 1 | 5 |
| 5 | What the data says (dots) | 0.5 | 5.5 |
| 6 | y = mx + b (the line) | 1 | 6.5 |
| 7 | Learning is finding m and b | 0.5 | 7 |
| 8 | More inputs, more m's | 0.5 | 7.5 |
| 9 | Squash it | 0.5 | 8 |
| 10 | Stack it | 1 | 9 |
| 10b | Three things under every model | 0.5 | 9.5 |
| 11 | The job for the next 32 minutes | 0.5 | 10 |
| 12 | Handoff | 0.5 | 10.5 |

Running long at slide 4: skip straight to slide 5. The definitions are where every AI talk
bleeds time.

---

## 1. Title

**On screen:** "Developing AI/ML. What it actually takes." Your name, role, date.

**Say:** One sentence on who you are and one on what the hour is: "I build and review this
stuff for a living. In the next hour you will watch a model get built, and then I will show
you why that was the easy part."

## 2. The word cloud

**On screen:** `figures/00-wordcloud.png`. Eighty terms squished together, mixed sizes.
Every term is coloured by the ring it belongs to on the definitions slide: blue for AI,
orange for machine learning, green for deep learning, yellow for data science, pink for
statistics, violet for agents. Nobody will notice until you tell them on slide 3, which is the
point. The terms and weights live in `tools/make_intro_figures.py`; edit there and re-run.

**Say:** Nothing for three seconds. Then: "Which of these has someone asked you about this
month?" Let two or three people answer. Do not read the slide.

## 3. One idea under all of them

**On screen:** The same cloud, faded to grey, with one line in the middle: **y = mx + b**,
and under it: *the equation, how you find its numbers, how you judge it, what you feed it,
what you wrap around it.*

**Say:** "Everything on that slide is one of five things. The equation you learned at
fourteen, how you find its numbers, how you judge it, what you feed it, or what you wrap
around it. And look at the colours: that mess is really six ideas. Two minutes to get them
out of the way."

## 4. Six colours, six ideas

**On screen:** A six-row table, one row per colour in the word cloud, each sentence phrased
on the no-show problem. The first three nest (deep learning inside machine learning inside
AI); the other three are what you do around a model.

| Term | The sentence on the slide |
|---|---|
| Artificial intelligence | The computer does something that looks like judgement, such as deciding which appointments are at risk. |
| Machine learning | It learned the rule from past appointments instead of a person writing the rule. |
| Deep learning | The same thing with many stacked layers. Needed for images, text, speech and signals. |
| Data science | Getting from raw records to a question a model can answer. Most of the work, and most of the time. |
| Statistics | The discipline of knowing how wrong you are. It shows up the moment we evaluate a model. |
| Agents, agentic AI | A model that can also take actions, like sending the reminder, and check what happened. Chain several toward a goal and people call it agentic. |

**Say:** One sentence each, no elaboration. Point at the colour in the cloud as you go if the
cloud is still on a second screen. The data science row is the one to lean on: "that row is
four of the six months." Hard stop at two and a half minutes. If someone wants to argue a
definition, offer to take it in Q&A.

## 4b. One problem for the whole hour

**On screen:** "Outpatient appointments that do not show up." Three bullets: the cost of an
empty chair, everyone has a theory, and the data: two years, twenty thousand appointments,
synthetic.

**Say:** "Here is the problem we will use for the rest of the hour. Every one of you has seen
it and most of you have a theory about it. The data is made up, on purpose, built to behave
like the real thing, so nothing on this screen is anyone's record. Before any of those six
definitions earns its keep, one picture of it."

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

**On screen:** `figures/04-weighted-sum.png`. Inputs down the left, one arrow each carrying
its own m, an "add it up, + b" node, and the no-show rate out the right. The equation sits
underneath in small type. This is a single neuron, though you do not say so yet.

**Say:** "Lead time is not the only thing that matters. Prior no-shows, whether a reminder went
out, how far they live. Each one gets its own arrow, and each arrow has its own m. The m's are
what people call weights. A model with a billion parameters is this picture with a billion
arrows." Point at the circle: "Everything funnels into one add-up. Remember the circle; it
comes back in two slides."

## 9. Squash it

**On screen:** `figures/03-squashed.png`. The straight line sailing past 100%, and the same
idea bent so it levels off.

**Say:** "A straight line has one problem. Book far enough out and it predicts a hundred and
ten percent of people no-show. So we bend it to stay between zero and one hundred. That bent
line is logistic regression, and it is the first model we train today."

## 10. Stack it

**On screen:** Rows, building down:
1. one squashed line (logistic regression)
2. several squashed lines feeding into another squashed line (a neural network)
3. many of those stacked (deep learning), with the words "images, text, speech, signals"
4. the same with a trick for attending to words in order (a large language model)
5. stacked if-thens instead of stacked lines, hundreds added up (trees: random forests,
   gradient boosting)

**Say:** "Stack a few of those bent lines so the output of one feeds the next, and you have a
neural network. Stack a lot, and you have deep learning. The large language models are this,
with a trick for paying attention to words in order, run on an enormous amount of text. It is
the same equation all the way down. The other popular family is trees: stacked if-thens
instead of stacked lines. Not the equation, but found by the same loop, measure how wrong and
take a step. In about twenty minutes you will watch a tree beat the neural network on our
data. Someone here was going to ask, so there it is."

## 10b. Three things under every model

**On screen:** A three-row table: algebra gives it a shape, calculus finds the numbers,
statistics says how wrong it is. One line under it: statistics is the one that tells you
whether to trust any of it. One more: some AI never learns at all.

**Say:** "Three things under every model you just saw. Algebra for the shape, calculus for
the finding, statistics for the judging. Statistics is the one that tells you whether to
trust the other two, and it is the second half of what you are about to watch. One caveat for
the purists: some AI never learns anything. Search, rules, the scheduling optimiser. Older,
still everywhere, and not today's subject."

Call it back twice: in block 4b, the hand-written loss function is where statistics enters the
loop, and the cell says so in a comment. At the top of block 5, say "everything from here is
the statistics row."

## 11. The job for the next 32 minutes

**On screen:** Four lines.
- Predict which outpatient appointments will be no-shows.
- The day before, for the scheduling team.
- Synthetic data, generated in front of you. No PHI anywhere.
- Three traps. Two in the data, one after launch.

**Say:** "Same problem, real code, and you can take it home and run it. Three traps: two in
the data, one after launch, the kind we find in real projects. I will show you each one when
the model falls into it."

## 12. Handoff

**On screen:** "Part 2: build it. 32 minutes." A screenshot of the notebook's first cell.

**Say:** "The loop cannot find m and b without data. So that is where we start, and it is
where real life starts too." Scroll to block 1.

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
