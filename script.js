function addNewRow(){
    console.log("Adding new row");
}

function addNewRow() {
    const table = document.getElementById("myTable");
    const newRow = table.insertRow();
    const cell1 = newRow.insertCell(0);
    const cell2 = newRow.insertCell(1);
    const cell3 = newRow.insertCell(2);

    // Set initial values for the new row
    cell1.innerHTML = table.rows.length - 1; // Update the row number
    cell2.innerHTML = '<input class="form-control form-control-sm w-25 text-center mx-auto" type="text" placeholder="">';
    cell3.innerHTML = '<input class="form-control form-control-sm w-25 text-center mx-auto" type="text" placeholder="">';
}

function deleteLastRow() {
    const table = document.getElementById("myTable");
    if (table.rows.length > 2) { // Ensure there are at least 2 rows (including spare row)
        table.deleteRow(table.rows.length - 1);
    }
}