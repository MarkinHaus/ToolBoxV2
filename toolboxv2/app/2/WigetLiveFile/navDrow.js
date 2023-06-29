document.getElementById('folder-opener').addEventListener('change', function(e) {
    var files = e.target.files; // FileList object

    // Clear the file list
    document.getElementById('file-list').innerHTML = '';

    // Loop through the FileList and render file names.
    for (var i = 0, f; f = files[i]; i++) {
        // Create a list item for each file and append it to the file list
        var listItem = document.createElement('li');
        listItem.textContent = f.name;
        listItem.addEventListener('click', function() {
            // Open the file and read its content
            var reader = new FileReader();
            reader.onload = function(e) {
                // Display the file content in the textarea
                document.getElementById('file-editor').value = e.target.result;
                document.getElementById('file-content').style.display = 'block';
            };
            reader.readAsText(f);
        });
        document.getElementById('file-list').appendChild(listItem);
    }
});

document.getElementById('save-button').addEventListener('click', function() {
    // Save the file content to localStorage
    var fileName = document.getElementById('file-list').querySelector('.active').textContent;
    var fileContent = document.getElementById('file-editor').value;
    localStorage.setItem(fileName, fileContent);
});
