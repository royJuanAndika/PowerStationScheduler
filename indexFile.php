<!DOCTYPE html>
<html>

<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>🦜 Polyglot - Piratical PyScript</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <link rel="stylesheet" href="https://pyscript.net/releases/2024.6.2/core.css">
    <script type="module" src="https://pyscript.net/releases/2024.6.2/core.js"></script>


    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@100;200;300;400;500;600;700;800;900&amp;display=swap" rel="stylesheet" />
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Quicksand:wght@300..700&display=swap" rel="stylesheet">
    <!-- Bootstrap icons-->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH" crossorigin="anonymous">
    <!-- Core theme CSS (includes Bootstrap)-->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="styles.css" rel="stylesheet" />
</head>

<body>

    <div class="navbar w-100 bg-blue">
        <div class="container-fluid d-flex flex-row">
            <div class="container w-25 d-flex justify-content-center">
                <img src="logo.png" alt="" style="height: 150px; width: 40%; object-fit: contain;">
                <div class="container d-flex align-items-center w-50 mx-0 px-0">
                    <h1 class="quicksandBold mt-2" style="color:white; font-size: 24px; display: inline-block;">Power Station Scheduler
                        <span class="quicksandThin" style="font-size: 17px;"> Genetic Algorithm</span></h1>
                </div>
            </div>
            <div class="container-fluid d-flex justify-content-center">
                
            </div>
        </div>
    </div>

    <div class="container-fluid d-flex justify-content-center align-items-center quicksandBold mt-5 px-5">
        <p class="w-75" style="font-size:20px;">Sebutkan power station yang negara anda miliki, berapa kapasitas listrik yang dihasilkan, dan interval yang dibutuhkan untuk maintenance. Kami akan membuatkan jadwal yang optimal untuk negara anda! </p>
    </div>


    <div class="container-fluid d-flex justify-content-center align-items-center quicksandBold mt-5 px-5 mb-5">
        <form action="upload.php" method="post" enctype="multipart/form-data">
            Select image to upload:
            <input type="file" name="fileToUpload" id="fileToUpload">
            <input type="submit" value="Upload Image" name="submit">
        </form>
    </div>


    <div class="container-fluid d-flex justify-content-center align-items-center quicksandBold mb-5">
        <button type="button" class="btn btn-outline-primary" style="width: 10%; font-size: 30px;" onclick="addNewRow()">+</button>
        <button type="button" class="btn btn-outline-danger" style="width: 10%; font-size: 30px;" onclick="deleteLastRow()">-</button>
    </div>
    
    <div class="container-fluid d-flex justify-content-center align-items-center quicksandBold px-5">
        
        <div class="input-group mb-3 w-25">
            <div class="input-group-prepend text-white">
                <span class="input-group-text" id="">Periode:</span>
            </div>
            <input type="text" class="form-control bg-dark text-white w-25" aria-label="Default" aria-describedby="inputGroup-sizing-default" value="6" id="formPeriode">
        </div><br>
        <div class="input-group mb-3 w-50">
            <div class="input-group-prepend text-white">
                <span class="input-group-text" id="">Kapasitas listrik minimal:</span>
            </div>  
            <input type="text" class="form-control bg-dark text-white w-25" aria-label="Default"
                aria-describedby="inputGroup-sizing-default" value="110" id="formMinListrik">
        </div>
        
        <div class="input-group mb-3 w-25">
            <div class="input-group-prepend text-white">
                <span class="input-group-text" id="">Engineer Teams Count:</span>
            </div>
            <input type="text" class="form-control bg-dark text-white w-25" aria-label="Default"
                aria-describedby="inputGroup-sizing-default" value="2" id="formEngineerTeamsCount">
        </div><br>

    </div>

    
    <div class="container-fluid d-flex justify-content-center align-items-center quicksandBold px-5 my-3">
        <div class="form-check rounded-left px-5 py-3" style="background-color:black;">
            <input class="form-check-input" type="radio" name="flexRadioDefault" id="flexRadioDefault1" value="elitism" checked>
            <label class="form-check-label" for="flexRadioDefault1" style="color: white;">
                Elitism Method
            </label>
        </div>  
        <div class="form-check rounded-right px-3 py-3" style="background-color:black;">
            <input class="form-check-input" type="radio" name="flexRadioDefault" id="flexRadioDefault2" value="tournament">
            <label class="form-check-label" for="flexRadioDefault2" style="color:white;">
                Tournament Method
            </label>
        </div>
    </div>
    </div>
        <div class="container-fluid d-flex justify-content-center align-items-center quicksandBold px-5 py-2">
            <button type="button" class="btn btn-outline-dark quicksandBold" style="font-size: 20px;" py-click="greet" onclick="accessValue()">
                <h1>Generate</h1>
            </button>
        </div>
    </div>

    

    <div class="container-fluid d-flex justify-content-center align-items-center quicksandBold px-5 w-75 py-2" id = "tableDiv">
        <table class="table table-dark w-25" id="shownTable" style="display:none">
            <thead>
                <tr id="tableTitle">
                </tr>
            </thead>
            <tbody id="tableTBody">
                <!-- <tr>
                    <th scope="row">1</th>
                    <td>Mark</td>
                    <td>Otto</td>
                    <td>@mdo</td>
                </tr>
                <tr>
                    <th scope="row">2</th>
                    <td>Jacob</td>
                    <td>Thornton</td>
                    <td>@fat</td>
                </tr> -->
        </table>

    </div>

    <div class="container-fluid d-flex justify-content-center align-items-center quicksandBold px-5 w-75 py-2" id="tableDiv" style="font-size:20px;">
        <h1 class="quicksandBold" id="runCount"></h1>
    </div>

    <script src="script.js"></script>
    <script type="py" src="./PowerStationSchedulerFile.py" config="./pyscript.json"></script>
</body>

<div class="container-fluid">
    <footer class="d-flex flex-wrap justify-content-between align-items-center border-top quicksandRegular">
        <p class="col-md-4 mb-0 text-muted" style="color:white;">&copy; 2024 Company, Inc</p>

        <a href="/"
            class="col-md-4 d-flex align-items-center justify-content-center mb-3 mb-md-0 me-md-auto link-dark text-decoration-none">

            <img src="logo.png" alt="" style="width: 20%; height:20%;">
        </a>

        <ul class="nav col-md-4 justify-content-end" style="color:white;">
            <li class="nav-item"><a href="#" class="nav-link px-2 text-muted" style="color:white;">Home</a></li>
            <li class="nav-item"><a href="#" class="nav-link px-2 text-muted">Features</a></li>
            <li class="nav-item"><a href="#" class="nav-link px-2 text-muted">Pricing</a></li>
            <li class="nav-item"><a href="#" class="nav-link px-2 text-muted">FAQs</a></li>
            <li class="nav-item"><a href="#" class="nav-link px-2 text-muted">About</a></li>
        </ul>
    </footer>
</div>


</html>