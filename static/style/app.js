const navSlide = () => {
    const burger = document.querySelector(".burger");
    const nav = document.querySelector(".nav-links");
    const navLinks = document.querySelectorAll(".nav-links li");

    burger.addEventListener("click", () => {
        // Toggle Nav
        nav.classList.toggle("nav-active");

        // Animate
        navLinks.forEach((link, index) => {
            if (link.style.animation) {
                link.style.animation = "";
            } else {
                link.style.animation = `navLinkFade 0.5 ease forwards ${index / 7 + 1.5
                    }s`;
                console.log(index / 7);
            }
        });
        // Burger Animation
        burger.classList.toggle('toggle');
    });
};
navSlide();

// Stop use Right Click
// document.addEventListener("contextmenu",function (e){
//     e.preventDefault();
// });



//For Selection Feild Data
function getSelectValue(){

    // Gender Data
    var gender  = document.getElementById("gender").value;
    console.log(gender);
    
    //Age
    var age  = document.getElementById("age").value;
    console.log(age);

    // Hypertension Data
    var hypertension  = document.getElementById("hypertension").value;
    console.log(hypertension);

    // Heart Disease
    var heart_disease  = document.getElementById("heart_disease").value;
    console.log(heart_disease);
    
    // ever_married
    var ever_married = document.getElementById("ever_married").value;
    console.log(ever_married);

    // Residence Type
    var Residence_type = document.getElementById("Residence_type").value;
    console.log(Residence_type);

    // Work Type
    var work_type  = document.getElementById("work_type").value;
    console.log(work_type);

     //Golucose Level
     var avg_glucose_level  = document.getElementById("avg_glucose_level").value;
     console.log(avg_glucose_level);
    
     //Golucose Level
     var bmi  = document.getElementById("bmi").value;
     console.log(bmi);
    
    // Somke
    var smoking_status  = document.getElementById("smoking_status").value;
    console.log(smoking_status);

}
getSelectValue();
