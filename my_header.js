document.addEventListener("DOMContentLoaded", function() {
    const header = document.querySelector('header h1');
    if (header) {
        const subHTML = `<sub><a href="https://www.linkedin.com/in/joshidevendra" target="_blank">Devendra Joshi</a></sub>`;
        header.insertAdjacentHTML('beforeend', subHTML);
    }
});
