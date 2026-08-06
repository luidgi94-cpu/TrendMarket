const search = document.getElementById("search");

search.addEventListener("keyup", function(){

const value = this.value.toLowerCase();

document.querySelectorAll(".card").forEach(card=>{

card.style.display =
card.innerText.toLowerCase().includes(value)
? "block"
: "none";

});

});