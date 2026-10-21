# Talk track: the 32-minute notebook walk-through

Whole-session shape for a talk that runs a little over an hour: the intro cells at the top of the
notebook (10, script in intro-slides.md), the demo blocks (32), "why it takes six months to ship" (15), seeded Q&A (10). About 67.

Audience: the technical breakout at an org-wide event. Developers, analysts, infra, informatics,
a few data-curious clinicians. Not a room of ML engineers. Narrate what each block does and
why, never the syntax.

Run every cell once before the session starts, then **Kernel > Restart Kernel and Clear
Outputs** so the room sees things appear. If anything dies, open
`notebooks/backup/no_show_classic_ml_prerun.ipynb` and keep going; the HTML beside it holds
every output for backup slides.

## Clock

| Block | Min | Cum. | What the room should take away |
|---|---|---|---|
| 1 Generate the data | 2 | 2 | The step that takes 20 seconds here and four months in real life |
| 2 Look at the data | 3 | 5 | The base rate. Plant the 82% number now |
| 3 Frame the prediction | 2 | 7 | Split by time, not at random |
| 4 Train a baseline | 3 | 10 | "Is 83% good?" |
| 4b What fit() is doing | 6 | 16 | Training is the nudging loop from the intro; deep is not automatically better |
| 5 Evaluate properly (live change 1) | 7 | 23 | The clinic picks the threshold, not the model |
| 6 What is the model using? (trap 1) | 4 | 27 | Target leakage. The data engineer is worth more than the model |
| 7 Does it work for everyone? (trap 2) | 3 | 30 | The model learned the process, not the patient |
| 8 Ship it, sort of (trap 3, live change 2) | 3 | 33 | Who is watching this number in month three? |

Running long: cut block 7 first, then the neural-network cell in 4b, then the ROC animation in 5. Blocks 5, 6 and 8 are
the talk.

## Block by block

**1. Generate the data.** Open with the callback: "This is the function that made every
picture in Part 1." Scroll the generator, do not read it. Point at the comment on
`reminder_confirmed` only if someone asks; otherwise it is the surprise for block 6.
Say: "This is the step that takes 20 seconds here and four months in real life."

**2. Look at the data.** Sample rows, then the rate cell prints 18%. Only then the talking
point: "everyone shows up" is 82% accurate. Plant it. Then the chart.

**3. Frame the prediction.** Predicting what, for whom, what happens with the answer. Then the
split by date. Say: "A random shuffle leaks the future into the past. Most tutorial models
cheat here without meaning to."

**4. Train a baseline.** Two models, seconds each. Accuracy prints at about 83%. Scroll to
the one-line cell under it, "Is that good?", and stop talking. Let it sit. Point back at 82%.

**4b. What fit() is doing.** Three cells, for the developers in the room. First, logistic
regression in ten lines of NumPy: predict, how wrong, which way is downhill, step, in a loop.
Read the four comments, not the code. Run the loop and show it lands on the same AUC as
scikit-learn to three decimals. Then the cell under it: the same loop on one input, slowed
down, drawing as it goes. Say nothing for the six seconds it runs. The line from the intro
starts flat and bends into place; the loss falls on the right. The m and b in the title are
in scaled units, so do not compare them to the intro's 4% and 7 points. Say: "That loop *is* training. It is the nudging from the
intro, and its name is gradient descent: the hill is the loss, and every step goes downhill.
Every model you will hear about this year, including the large language models, is some
version of those four lines run a few billion times." Second, the same loop in PyTorch; point
at `loss.backward()` as the only new idea: it works out which way is downhill for you. Third, a small neural network on the
same data. It loses to the boosted tree, and its training loss is lower than the linear
model's while its test score is worse. Say: "It memorised the training set. Pick the model for
the data, not for the press release." Cut the third cell if running long.

**5. Evaluate properly.** Count true and false positives by hand first, two divisions, and
show accuracy from block 4 falling out of the same four numbers. Then the library report and the
confusion matrix. **ROC animation (1 min):** the threshold line slides down the score histogram and
each stop becomes one dot on the right; the dots are the ROC curve, the shading is the AUC. Say:
"The curve is the threshold sweep, drawn. Every dot is a clinic policy." The by-hand area matches
`roc_auc_score`, and the last line gives AUC its meaning: pick a real no-show and a real show-up at
random, 0.76 is how often the model ranked them the right way round. "It measures how well the model
sorts, not how well it decides." Then the library curves and the sweep table. **Live change 1:** edit `THRESHOLD`, re-run. Try 0.15 (catches three quarters, flags
nearly half the clinic) then 0.50 (flags 6%, catches one in five).
Say: "The model does not decide the threshold. The clinic does. Reminder call, go low.
Overbooking the slot, go high."

**6. What is the model using? (trap 1).** Show the honest importance chart: lead time, prior no-shows,
reminders. Then: "Someone from the front-desk system says you skipped a column." Add
`reminder_confirmed`, retrain. Accuracy jumps to 96% and the importance chart is one orange
bar. Say: "That field is written at check-in. It is populated after the outcome. The model
found the answer key." Name it: target leakage. "The data engineer who knows how that field
gets populated is worth more than the model."

**7. Does it work for everyone? (trap 2).** Recall by portal account. The group with the *higher* no-show rate is
the group the model catches *least*, roughly half the recall. Say: "The reminder process
reached them less and the system recorded their history worse. The model learned the process,
not the patient. Ship this and the reminder calls go to the people already getting reminders."

**8. Ship it, sort of.** Save, load, `predict_one` twice. Then **trap 3, live change 2:** the drift
batch with `lead_time_mean=45`. Share flagged jumps from 17% to 45%. Show the histogram.
Say: "Nothing is broken. No alert fired. The endpoint returns 200. This is month three.
Who is watching this number?" That line is the handoff to the governance section.

## Seeded Q&A

- "What would you want to automate first?"
- "Which column in your system is the `reminder_confirmed`?"

## Classic ML vs LLM for the demo

Classic ML (this notebook) shows the whole loop, is deterministic, and keeps PHI questions out
of the demo. If the event framing is explicitly generative AI, keep this spine and add a
five-minute LLM contrast at the end: same no-show problem, a prompt asking a model to score
one appointment, and three points: no training loop, no threshold you control, and every
appointment you send is data leaving the building. Do not try to make an LLM the 25 minutes.
