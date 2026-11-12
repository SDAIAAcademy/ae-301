# Migration of applied-nlp

This course was moved from the previous slide system by `academy migrate`. The inner markup of every slide is unchanged; each `<section>` tag got an `id`, section slides a session number, lab slides their lab id, and a slide drawn with coordinates or holding a layout of the course's own stylesheet that cannot reflow `data-read="scaled"`. Each deck keeps its old file name as its unit id, so its published address keeps answering. `labs/` and `assets/` are copied as they were.

Run the engine from this folder with `node academy.mjs <command>` (`check`, `build`, `check --layout --shots`); it finds the engine in `$ACADEMY_ENGINE`, then the folder named in `.academy-engine`, then `../engine`. Or run `node <engine>/bin/academy.mjs <command> --root <this folder>`.

## What is left, and how long it takes

About **1 h 7 min** of a person's work before `academy check` passes, and about **3 h 28 min** more to clear its warnings, which the house standard also asks for. The estimate counts each task below and each finding of `academy check` as this migration found them; a coding agent does the same work faster, but a person still decides the outcomes of each session and reviews the checkpoints.

| Work | Minutes |
|---|---|
| The tasks under "Needs a person" (5) | 59 |
| 3 × `a11y/svg` (error) | 6 |
| 1 × `text/placeholder` (error) | 2 |
| 4 × `notes/ratio` (warning) | 138 |
| 9 × `quiz/why` (warning) | 45 |
| 5 × `slide/dense` (warning) | 25 |

Then `node academy.mjs check --layout --shots` opens every slide in Chrome (a few minutes for a course of five decks). It reports what the static check cannot see: a slide whose content is cut off or wider than the slide (`layout/overflow`, about 5 minutes each; long lines of code are the usual case, and the old system's own `?check` reported most of them too) and slides that do not reflow in the reading view on a phone (`layout/phone-wide`, `layout/phone-text`). Compare its contact sheets (`.shots/<lang>/<deck>/sheet-<n>.png`) with the old decks.

## Slides

Counted in the old deck and in the new source; "a → b" where they differ.

| Old deck | New deck | Slides | Speaker notes | Checkpoints | Lab slides | Pictures | Fragments | Break slides dropped |
|---|---|---|---|---|---|---|---|---|
| slides/day1.html | slides/day1.html | 53 | 4 | 3 | 4 | 3 | 0 | 0 |
| slides/day2.html | slides/day2.html | 47 | 1 | 3 | 4 | 1 | 0 | 0 |
| slides/day3.html | slides/day3.html | 43 | 2 | 3 | 4 | 2 | 0 | 0 |
| slides/day4.html | slides/day4.html | 20 | 3 | 0 | 0 | 0 | 0 | 0 |
| All | | 163 | 10 | 9 | 12 | 6 | 0 | 0 |

## Rebuilt from the old files

- "extensions" in course.json: assets/migrated.css, which the old decks loaded after the system's own files, so the course's own classes keep their look.
- slides/day1.html → slides/day1.html: "Text Representation & Transformer Foundations", 4 sessions, 4 labs.
- slides/day2.html → slides/day2.html: "LLMs & Applied NLP Systems", 4 sessions, 4 labs.
- slides/day3.html → slides/day3.html: "Fine-Tuning & NLP Engineering", 4 sessions, 4 labs.
- slides/day4.html is the capstone day (no lab slides; presentation slides or a capstone title): its sessions are not sessions in the new engine, so data-session was removed from its section slides.
- 6 course outcomes from the list of outcomes on index.html (LO1 to LO6).
- The audience, from the "Audience" paragraph of index.html.
- 1 prerequisite from the "Environment" paragraph of index.html.
- The capstone summary, from the "Capstone" section of index.html (its paragraphs and its list of options).
- The capstone submission, from the paragraph of the "Capstone" section that starts with "Submit".
- The capstone rubric: Functional end-to-end solution 30, Model selection & implementation 25, Evaluation & error analysis 20, Fine-tuning or justified adaptation 15, Presentation & reproducibility 10.
- The capstone's talk_minutes (10) and qa_minutes (5), from the timers "Talk + demo" and "Q&A" of the old slides/day4.html line 158.
- Title "Applied NLP Engineering", track ai-engineering, language en; the id "applied-nlp" comes from the folder name.

## Needs a person

- assets/migrated.css holds the <style> rules of the old slides/day4.html line 9, which now apply to every deck. Look at those slides in the contact sheets of "academy check --shots"; move the rules into assets/course.css or delete the ones the engine's theme already gives. (about 10 min)
- slides/day3.html line 505: the checklist of lab 3.3 adds up to 24 minutes, the lab is 25; the steps were left out of course.json until they agree. Change one step's minutes on the lab slide (the check reports it as lab/checklist), then copy the steps into course.json. (about 5 min)
- Which outcomes each session serves: write "lo" for each of the 12 sessions in course.json (["LO1"], or more than one), from the session titles and the slides; the old decks do not say, so until then "academy check" reports every outcome as not covered (lo/uncovered). (about 24 min)
- The shape: the decks do not follow the house format everywhere (four sessions a day, a 25-minute lab in each), so course.json states the course's own rules in "checks": {"sessions_per_day":4,"session_lab":true,"lab_minutes":null}. Decide whether the course should move to the house format. (about 10 min)
- 1 part of index.html that the migration does not read (listed under "Left out of index.html"): keep or drop each. (about 10 min)

## What `academy check` reports

On the course as migrated, apart from the codes a task above clears (`lab/checklist`, `lo/uncovered`). `node academy.mjs check` prints each place with what to do.

| Code | Level | Findings | Minutes | First places |
|---|---|---|---|---|
| `a11y/svg` | error | 3 | 6 | slides/day1.html:626, slides/day3.html:298, slides/day3.html:400 |
| `text/placeholder` | error | 1 | 2 | slides/day3.html:121 |
| `notes/ratio` | warning | 4 | 138 | slides/day1.html, slides/day2.html, slides/day3.html … |
| `quiz/why` | warning | 9 | 45 | slides/day1.html:339, slides/day1.html:557, slides/day1.html:775 … |
| `slide/dense` | warning | 5 | 25 | slides/day1.html:193, slides/day1.html:835, slides/day2.html:259 … |

## Left out of index.html

- index.html line 120: "Setup", the paragraph "Stack" (1 paragraph) is not on the new course page, which is generated from course.json. Move what is still true into course.json (a "sources" group for a list of links, "prerequisites" for what to prepare), README.md or a lab page, or drop it.

## Good to know

- 9 checkpoint slides written as <section class="slide quiz"> (slides/day1.html line 339, slides/day1.html line 557, slides/day1.html line 775, slides/day2.html line 235, slides/day2.html line 441, slides/day2.html line 615, slides/day3.html line 198, slides/day3.html line 353, slides/day3.html line 524) are kept as written. Each is one activity, in the build and in the browser: its lead paragraph (p.lead), or else its heading, is the question; the room answers with the number keys or a click. "academy check" asks for the misconception behind each wrong option (quiz/why).

## Break slides dropped

- None.

## Markup that does not parse strictly

The build parses slide sources strictly. Fix each place below in the slide source (close the element, quote the attribute, write `&lt;` for a "<" in text).

- None: every deck parses strictly.
