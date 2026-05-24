function copyBibtex() {
  const text = document.getElementById("bibtex").textContent.trim();
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById("copy-btn");
    const original = btn.textContent;
    btn.textContent = "Copied!";
    setTimeout(() => { btn.textContent = original; }, 2000);
  });
}
