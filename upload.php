<?php
$target_dir = "/home/pyodide"; // Specify the directory where the file will be saved
$target_file = $target_dir . basename($_FILES["fileToUpload"]["name"]);
$uploadOk = 1;
$imageFileType = strtolower(pathinfo($target_file, PATHINFO_EXTENSION));

// Check if the file is an actual image or fake image
if (isset($_POST["submit"])) {
    // $check = getimagesize($_FILES["fileToUpload"]["tmp_name"]);
    // if ($check !== false) {
    //     echo "File is an image - " . $check["mime"] . ".";
    //     $uploadOk = 1;
    // } else {
    //     echo "File is not an image.";
    //     $uploadOk = 0;
    // }
    $uploadOk = 1;
}

// Move the uploaded file to the desired location
if ($uploadOk) {
    move_uploaded_file($_FILES["fileToUpload"]["tmp_name"], $target_file);
    
    echo "File is uploaded.";
    // Additional processing (e.g., resizing, renaming) can be done here
}
?>