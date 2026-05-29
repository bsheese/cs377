"use strict";

const app = document.getElementById("app");
let UNITS = [];

// ---- rendering helpers -------------------------------------------------
function shuffle(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// Register the math-aware tokenizer once, so `$...$` is parsed as math by
// marked itself (rather than scanned after the fact). This keeps marked from
// treating `_`/`*` inside math as emphasis, and renders KaTeX inline.
if (window.markedKatex) {
  marked.use(markedKatex({ throwOnError: false, nonStandard: false }));
}

// Render inline Markdown + LaTeX into an element.
function renderRich(el, text) {
  el.innerHTML = marked.parseInline(text);
}

function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  Object.assign(node, props);
  for (const c of [].concat(children)) {
    if (c != null) node.append(c);
  }
  return node;
}

// ---- picker view -------------------------------------------------------
function showPicker() {
  location.hash = "";
  app.innerHTML = "";
  for (const unit of UNITS) {
    const sections = el("div", { className: "sections" });
    unit.sections.forEach((sec, i) => {
      const count = el("span", {
        className: "count",
        textContent: `${sec.questions.length} questions`,
      });
      const btn = el("button", { className: "section-btn" }, [
        el("span", { textContent: sec.title }),
        count,
      ]);
      btn.onclick = () => startQuiz(unit, i);
      sections.append(btn);
    });
    app.append(
      el("section", { className: "unit" }, [
        el("h2", { textContent: unit.title }),
        sections,
      ])
    );
  }
}

// ---- quiz view ---------------------------------------------------------
function startQuiz(unit, sectionIdx) {
  const section = unit.sections[sectionIdx];
  const questions = shuffle(section.questions).map((q) => ({
    ...q,
    shuffled: shuffle(q.options),
  }));
  const answers = new Array(questions.length).fill(null);
  let index = 0;

  function render() {
    app.innerHTML = "";
    if (index >= questions.length) return showResults();

    const q = questions[index];
    const chosen = answers[index];
    const locked = chosen !== null;

    const back = el("a", {
      className: "back-link",
      href: "#",
      textContent: "← All quizzes",
    });
    back.onclick = (e) => {
      e.preventDefault();
      showPicker();
    };

    app.append(
      el("div", { className: "quiz-head" }, [
        el("strong", { textContent: section.title }),
        el("span", {
          className: "progress",
          textContent: `Question ${index + 1} of ${questions.length}`,
        }),
      ]),
      back
    );

    const qEl = el("div", { className: "question" });
    renderRich(qEl, q.question);
    app.append(qEl);

    const opts = el("div", { className: "options" });
    q.shuffled.forEach((opt) => {
      const input = el("input", {
        type: "radio",
        name: "opt",
        value: opt,
        checked: chosen === opt,
        disabled: locked,
      });
      const label = el("span");
      renderRich(label, opt);
      const row = el("label", { className: "option" }, [input, label]);
      if (locked) {
        row.classList.add("disabled");
        if (opt === q.answer) row.classList.add("correct");
        else if (opt === chosen) row.classList.add("wrong");
      }
      opts.append(row);
    });
    app.append(opts);

    const feedback = el("div", { className: "feedback" });
    if (locked) {
      if (chosen === q.answer) {
        feedback.classList.add("correct");
        feedback.textContent = "✅ Correct!";
      } else {
        feedback.classList.add("wrong");
        const ans = el("span");
        renderRich(ans, q.answer);
        feedback.append("❌ Incorrect. Correct answer: ");
        feedback.append(ans);
      }
    }
    app.append(feedback);

    const backBtn = el("button", {
      className: "nav",
      textContent: "Back",
      disabled: index === 0,
    });
    backBtn.onclick = () => {
      index--;
      render();
    };

    const submitBtn = el("button", {
      className: "nav primary",
      textContent: "Submit",
      disabled: locked,
    });
    submitBtn.onclick = () => {
      const sel = opts.querySelector("input:checked");
      if (!sel) {
        feedback.classList.remove("correct");
        feedback.classList.add("wrong");
        feedback.textContent = "⚠️ Please select an option.";
        return;
      }
      answers[index] = sel.value;
      render();
    };

    const isLast = index === questions.length - 1;
    const nextBtn = el("button", {
      className: "nav",
      textContent: isLast ? "See results" : "Next",
      disabled: !locked,
    });
    nextBtn.onclick = () => {
      index++;
      render();
    };

    app.append(el("div", { className: "controls" }, [backBtn, submitBtn, nextBtn]));
  }

  function showResults() {
    const score = answers.reduce(
      (n, a, i) => n + (a === questions[i].answer ? 1 : 0),
      0
    );
    const pct = ((score / questions.length) * 100).toFixed(1);

    const retry = el("button", {
      className: "nav primary",
      textContent: "Retake",
    });
    retry.onclick = () => startQuiz(unit, sectionIdx);
    const home = el("button", { className: "nav", textContent: "All quizzes" });
    home.onclick = showPicker;

    app.innerHTML = "";
    app.append(
      el("div", { className: "results" }, [
        el("h2", { textContent: "Quiz finished" }),
        el("div", { className: "score", textContent: `${pct}%` }),
        el("div", {
          className: "detail",
          textContent: `You scored ${score} out of ${questions.length}`,
        }),
        el("div", { className: "actions" }, [retry, home]),
      ])
    );
  }

  render();
}

// ---- boot --------------------------------------------------------------
fetch("./quizzes.json")
  .then((r) => {
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  })
  .then((data) => {
    UNITS = data;
    showPicker();
  })
  .catch((err) => {
    app.textContent = `Could not load quizzes: ${err.message}`;
  });
