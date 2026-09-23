# Project Summary (Plain-Language Version)

## What problem are we trying to solve?

Satellites can record video of the ground — cars, ships, planes moving around. Software then tries to draw a box around each moving object and keep track of it, frame by frame, giving it an ID number so we know "this is the same car as three seconds ago."

The hard part is **occlusion**: when a car drives under a bridge, or behind a building, it disappears from view for a few seconds. When it comes back out, does the software recognize it as the *same* car, or does it think it's a brand-new one? Losing track of identity like this is one of the biggest weaknesses in this kind of system.

## Our idea

When a satellite films the ground, the satellite itself is moving very fast. That means the whole picture drifts and shifts slightly from frame to frame — even things that aren't moving (buildings, roads) appear to shift a little, just because the camera is moving. Most existing tracking software doesn't separate out "the picture shifted because the satellite moved" from "the car actually moved." It just watches the raw video and guesses where the car will reappear.

**Our idea:** if the software first mathematically works out how much the picture shifted due to the satellite's own movement, and removes that shift, then predicting where a hidden car will reappear should become more accurate — because the prediction is now based only on the car's *real* motion, not a mix of real motion and camera drift.

## What we actually built

Not just the idea on paper — a full working system:

- A program that watches two video frames and automatically works out how much the camera itself moved between them (no GPS or satellite orbit data needed — it figures this out purely from the images).
- A tracker that uses this information to follow multiple objects at once, predict where they'll reappear after being hidden, and give them back their correct ID number when they do.
- A website-style dashboard where you can start a test, watch it run live (frame by frame, with the reasoning steps visualized), and browse the results of every past test.
- Tools to load real satellite video datasets (VISO and SDM-Car) that researchers have published for exactly this kind of testing.

## The honest story of how testing went

This part matters because it shows the testing was done carefully, not just accepted at face value:

1. **First tests ran, but the numbers looked wrong.** We found and fixed several real mistakes in the code — for example, a setting meant to control "how long to wait before giving up on a hidden object" wasn't actually connected to anything, so it was silently ignored.
2. **We discovered a surprise:** the published satellite videos we were testing on barely show any camera drift at all — the pictures are almost perfectly steady. That's actually a known issue other researchers in this field have run into too. It meant our idea had nothing real to prove itself against, since there was barely any camera drift to correct for.
3. **So we built a way to artificially add a controlled, realistic amount of camera drift** to the test videos, so we could properly test whether removing it helps. This is a standard, accepted way to test this kind of idea when real footage doesn't have enough of the effect you're studying.
4. **We also found that our own "how many times did the system lose and recover an object" counter was wildly overcounting** — it was counting ordinary detection hiccups as if they were real occlusion events. We built a stricter, fairer way of counting that only credits the system with a "recovery" when it's recovering from a real, verified hidden period (confirmed against the dataset's own official labels).
5. **After fixing all of that**, we ran a first proper test on one video. Result: **the camera-drift-correcting version recovered the correct identity 5 times out of 14 real hidden-object cases; the version without that correction recovered it only 3 times out of 14.** That's a promising direction — but 14 cases is a very small number to draw a firm conclusion from.

## Where things stand right now

We built an automated system that runs this test across **many different videos and many random variations**, so instead of one small comparison, we get a large batch of results, and a proper statistical check on whether the pattern we're seeing is real or just chance. That batch is currently running (and can be safely paused and resumed without losing progress, so it doesn't have to run all at once).

## Is this ready to publish or claim as a proven result?

**Not yet, and that's completely normal at this stage.** Specifically still missing:

- All testing so far uses "perfect" object detection (the software is told exactly where each object is, taken from the dataset's official answer key) — this was deliberate, to test the *tracking* idea on its own, but a real system needs a real detector that finds objects itself, and that hasn't been plugged in yet.
- We haven't yet confirmed that our artificially-added camera drift behaves like a real satellite's drift would.
- Only one video has produced a full, trustworthy result so far; the bigger multi-video test is what will actually tell us if this holds up.
- We haven't yet compared our system against other published tracking methods — right now it's only being compared against itself with the camera-drift-correction turned on vs. off.

## Bottom line

The core idea has passed its first real test in a promising direction, and the testing process itself has been rigorous — every shortcut or possible source of error found so far has been investigated and fixed rather than ignored. The next step is simply to gather enough test results, across enough different videos, to know with confidence whether the promising early signal holds up.
