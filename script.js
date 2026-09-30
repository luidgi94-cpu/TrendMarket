const search = document.getElementById("search");

if (search) {
  search.addEventListener("input", function () {
    const value = this.value.trim().toLowerCase();

    document.querySelectorAll(".card").forEach(card => {
      const match = card.innerText.toLowerCase().includes(value);
      card.style.display = match ? "" : "none";
    });
  });
}
