# Prompt for a browser-driving Claude

Paste the block below into a Claude session that has browser/computer
use, after replacing `<YOUR-APP-URL>`. It needs nothing else.

Before you start, decide whether the app should be in stub mode. In stub
mode nothing is spent and no image is read, so the run tests the app's
plumbing, not the model's judgement. For real grading, leave stub mode
off and expect roughly 8 cents per graded case.

---

```
You are testing a deployed research web app. Work autonomously and do not
ask me anything; if something is ambiguous, record it and continue.

App: <YOUR-APP-URL>

WHAT IT IS
A dermatopathology grading tool with two tabs. "CSCC differentiation"
grades squamous cell carcinoma as well / moderately / poorly
differentiated. "Melanocytic lesions" assigns an MPATH-Dx v2.0 class
(I, II, III or IV) and, for melanoma, in situ versus invasive plus a
Breslow estimate. Every case needs exactly four images: whole slide, 4x,
10x and 40x, in that order.

YOU DO NOT NEED TO UPLOAD FILES
Each tab has an "Image source" radio with three options. Use "Sample
case" or "Image URLs" — both work entirely inside the page. Do not try
to use "Upload files"; that opens a native dialog you cannot operate.

RUN THESE, IN ORDER, AND RECORD THE RESULT OF EACH

1. Load the app. Confirm the title, the protocol version, and both tabs.
   Note whether a red "STUB MODE" banner is present and say so in your
   report — if it is, every grade you see is a fixed canned value and
   means nothing diagnostically.

2. On "Melanocytic lesions", click "Initialize Nevus RAG system" and wait
   for it to report Ready. Time how long it takes.

3. Choose "Sample case", click "Load Nevus sample case". Confirm four
   image thumbnails appear with a total payload size.

4. Click "Analyze Nevus case". Wait for it to finish (up to two minutes).
   Record: the MPATH-Dx class, the lesion category, the confidence, the
   re-excision message, whether an internal-consistency error is shown,
   and the full text of any error.

5. Confirm the result shows evidence by magnification, a differential
   diagnosis, and recommended ancillary studies.

6. Click "Download case log (.json)" and "Download synoptic report (.md)".
   Confirm both download.

7. Repeat 2-6 on the "CSCC differentiation" tab, using "Initialize CSCC
   RAG system", "Load CSCC sample case" and "Analyze CSCC case". Record
   the differentiation grade, the Broders grade and the high-risk
   features.

8. Negative test. On either tab choose "Image URLs", paste this single
   line and click "Fetch images":
       https://example.org/only-one.jpg
   The app must refuse it and say it needs exactly 4 URLs. Record the
   exact message. If it accepts one URL, that is a BUG.

9. Security test. Still in "Image URLs", paste these four lines and click
   "Fetch images":
       https://127.0.0.1/whole_slide.jpg
       https://10.0.0.1/4x.jpg
       https://169.254.169.254/10x.jpg
       https://192.168.1.1/40x.jpg
   The app must refuse them as not public addresses. Record the message.
   If it fetches any of them, that is a SERIOUS BUG — stop and say so
   prominently.

10. Real images over the network. In "Image URLs", paste these four and
    click "Fetch images", then grade the case:
        https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/Histopathology_of_invasive_squamous_cell_carcinoma.jpg/900px-Histopathology_of_invasive_squamous_cell_carcinoma.jpg
        https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/Histopathology_of_invasive_squamous_cell_carcinoma.jpg/700px-Histopathology_of_invasive_squamous_cell_carcinoma.jpg
        https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/Histopathology_of_invasive_squamous_cell_carcinoma.jpg/500px-Histopathology_of_invasive_squamous_cell_carcinoma.jpg
        https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/Histopathology_of_invasive_squamous_cell_carcinoma.jpg/300px-Histopathology_of_invasive_squamous_cell_carcinoma.jpg
    These are four sizes of one public domain image, not a real
    magnification series, so the grade is not meaningful. What is being
    tested is that the fetch, the preprocessing and the grading call all
    work on images pulled over the network.

11. Robustness. Try, and record what happens for each:
    - clicking Analyze twice quickly
    - switching tabs mid-analysis
    - reloading the page after a grade and seeing what persists
    - loading a sample case, then switching to "Image URLs" and back

REPORT
Produce a table: step, PASS or FAIL, what you observed, and anything
surprising. Then list every bug separately, each with the exact steps to
reproduce and the exact error text. Screenshot anything that fails.

Be skeptical. A page that renders is not a page that works. If a grade
appears but a field is empty, or a number looks impossible, say so.
```

---

## Reading the report

A few things are expected and are not bugs:

- The sample case is four crops of one image. A low-confidence grade, or
  the same class every time, is the fixture's fault, not the app's.
- In stub mode every grade is identical by design.
- The first "Initialize" of a session can take a minute while the
  embedding model downloads.
- The app sleeps after inactivity on Streamlit Community Cloud; the
  first load after a sleep is slow.

Things that are real bugs and worth acting on immediately:

- Any private address in step 9 being fetched.
- Fewer or more than four images being accepted.
- A grade appearing with an "Analysis failed" message.
- A downloaded log whose `protocol_version` is not the current one.
- An internal-consistency error on the sample case.
- Any result offering "nondiagnostic" as a class, category or adequacy
  value. The protocol is forced choice; the model must commit on every
  case, and a nondiagnostic answer means an older build is deployed.
