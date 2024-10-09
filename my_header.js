document.addEventListener("DOMContentLoaded", function() {
    const headerHTML = `
        <header>
            <h1>Your Blog Title<sub><a href="https://www.linkedin.com/in/yourprofile" target="_blank">Your Name</a></sub></h1>
        </header>
    `;
    document.body.insertAdjacentHTML('afterbegin', headerHTML);
});
