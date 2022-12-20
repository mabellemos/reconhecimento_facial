var paragrafo1 = document.querySelector('#paragrafo1');
var texto1 = document.querySelector('#paragrafo1').innerText;
var flork_chinela = document.getElementById('flork');
var flork_chorando = document.getElementById('flork');
var flork_feliz = document.getElementById('flork');
let botao1 = document.querySelector('.botao1')
botao1.addEventListener(click, feliz);
let botao2 = document.querySelector('.botao2');
botao2.addEventListener(click, triste);

writer = new Typewriter(paragrafo1, {
    //loop: true,
});

var type
typewriter
    .typeString(texto1)
    .pauseFor(1500)
    .start()
    
function feliz(){
    document.getElementById('flork_feliz').style.display = 'block';
}

function triste(){
    document.getElementById('flork_chinela').style.display = 'block';
}