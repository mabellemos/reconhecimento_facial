var paragrafo1 = document.querySelector('#paragrafo1');
var texto1 = document.querySelector('#paragrafo1').innerText;
writer = new Typewriter(paragrafo1, {
    //loop: true,
});

var type
typewriter
    .typeString(texto1)
    .pauseFor(1500)
    .start()