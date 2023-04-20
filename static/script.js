// ---------Responsive-navbar-active-animation-----------
function test(){
	var tabsNewAnim = $('#navbarSupportedContent');
	var selectorNewAnim = $('#navbarSupportedContent').find('li').length;
	var activeItemNewAnim = tabsNewAnim.find('.active');
	var activeWidthNewAnimHeight = activeItemNewAnim.innerHeight();
	var activeWidthNewAnimWidth = activeItemNewAnim.innerWidth();
	var itemPosNewAnimTop = activeItemNewAnim.position();
	var itemPosNewAnimLeft = activeItemNewAnim.position();
	$(".hori-selector").css({
		"top":itemPosNewAnimTop.top + "px", 
		"left":itemPosNewAnimLeft.left + "px",
		"height": activeWidthNewAnimHeight + "px",
		"width": activeWidthNewAnimWidth + "px"
	});
	$("#navbarSupportedContent").on("click","li",function(e){
		$('#navbarSupportedContent ul li').removeClass("active");
		$(this).addClass('active');
		var activeWidthNewAnimHeight = $(this).innerHeight();
		var activeWidthNewAnimWidth = $(this).innerWidth();
		var itemPosNewAnimTop = $(this).position();
		var itemPosNewAnimLeft = $(this).position();
		$(".hori-selector").css({
			"top":itemPosNewAnimTop.top + "px", 
			"left":itemPosNewAnimLeft.left + "px",
			"height": activeWidthNewAnimHeight + "px",
			"width": activeWidthNewAnimWidth + "px"
		});
	});
}
$(document).ready(function(){
	setTimeout(function(){ test(); });
});
$(window).on('resize', function(){
	setTimeout(function(){ test(); }, 500);
});
$(".navbar-toggler").click(function(){
	$(".navbar-collapse").slideToggle(300);
	setTimeout(function(){ test(); });
});


jQuery(document).ready(function($){
	var path = window.location.pathname.split("/").pop();

	if ( path == '' ) {
		path = 'index.html';
	}
	var target = $('#navbarSupportedContent ul li a[href="'+path+'"]');
	target.parent().addClass('active');
});

var midCol = document.getElementById('midCol'), 
childDiv = [midCol.getElementsByClassName('drag-area')[0], midCol.getElementsByClassName('drag-area')[1], midCol.getElementsByClassName('drag-area')[2]],
dragText1 = childDiv[0].querySelector("header"), dragText2 = childDiv[1].querySelector("header"), dragText3 = childDiv[2].querySelector("header");
const fileName1 = childDiv[0].querySelector(".file-name"), fileName2 = childDiv[1].querySelector(".file-name"), fileName3 = childDiv[2].querySelector(".file-name");
const defaultBtn1 = childDiv[0].querySelector("#default-btn"), defaultBtn2 = childDiv[1].querySelector("#default-btn"), defaultBtn3 = childDiv[2].querySelector("#default-btn");
const cancelBtn1 = childDiv[0].querySelector("#cancel-btn i"), cancelBtn2 = childDiv[1].querySelector("#cancel-btn i"), cancelBtn3 = childDiv[2].querySelector("#cancel-btn i");
const img1 = childDiv[0].querySelector("#uploadedImg"), img2 = childDiv[1].querySelector("#uploadedImg"), img3 = childDiv[2].querySelector("#uploadedImg");
const imgDiv1 = childDiv[0].querySelector(".drag-area .image"), imgDiv2 = childDiv[1].querySelector(".drag-area .image"), imgDiv3 = childDiv[2].querySelector(".drag-area .image");
let file1, file2, file3;
let regExp = /[0-9a-zA-Z\^\&\'\@\{\}\[\]\,\$\=\!\-\#\(\)\.\%\+\~\_ ]+$/;
function defaultBtnActive1(){
	defaultBtn1.click();
}
function defaultBtnActive2(){
	defaultBtn2.click();
}
function defaultBtnActive3(){
	defaultBtn3.click();
}

//Event listener for image upload dialog 1
defaultBtn1.addEventListener("change", function(){
const file1 = this.files[0];
if(file1){
	const reader = new FileReader();
	reader.onload = function(){
	const result = reader.result;
	img1.src = result;
	imgDiv1.setAttribute('style', 'display:block !important');
	childDiv[0].classList.add("active");
	}
	cancelBtn1.addEventListener("click", function(){
		img1.src = "";
		imgDiv1.setAttribute('style', 'display:none !important');
		childDiv[0].classList.remove("active");
	})
	reader.readAsDataURL(file1);
}
if(this.value){
	let valueStore = this.value.match(regExp);
	fileName1.textContent = valueStore;
}
});

//Event listener for image upload dialog 2
defaultBtn2.addEventListener("change", function(){
const file2 = this.files[0];
if(file2){
	const reader = new FileReader();
	reader.onload = function(){
	const result = reader.result;
	img2.src = result;
	imgDiv2.setAttribute('style', 'display:block !important');
	childDiv[1].classList.add("active");
	}
	cancelBtn2.addEventListener("click", function(){
		img2.src = "";
		imgDiv2.setAttribute('style', 'display:none !important');
		childDiv[1].classList.remove("active");
	})
	reader.readAsDataURL(file2);
}
if(this.value){
	let valueStore = this.value.match(regExp);
	fileName2.textContent = valueStore;
}
});

//Event listener for image upload dialog 3
defaultBtn3.addEventListener("change", function(){
const file3 = this.files[0];
if(file3){
	const reader = new FileReader();
	reader.onload = function(){
	const result = reader.result;
	img3.src = result;
	imgDiv3.setAttribute('style', 'display:block !important');
	childDiv[2].classList.add("active");
	}
	cancelBtn3.addEventListener("click", function(){
		img3.src = "";
		imgDiv3.setAttribute('style', 'display:none !important');
		childDiv[2].classList.remove("active");
	})
	reader.readAsDataURL(file3);
}
if(this.value){
	let valueStore = this.value.match(regExp);
	fileName3.textContent = valueStore;
}
});

//It changes the text when user drags over the specified div element
childDiv[0].addEventListener("dragover", (event)=>{
  event.preventDefault();
  childDiv[0].classList.add("active");
  dragText1.textContent = "Release to Upload File";
});

childDiv[1].addEventListener("dragover", (event)=>{
	event.preventDefault();
	childDiv[1].classList.add("active");
	dragText2.textContent = "Release to Upload File";
  });

childDiv[2].addEventListener("dragover", (event)=>{
	event.preventDefault();
	childDiv[2].classList.add("active");
	dragText3.textContent = "Release to Upload File";
});

//It changes the text when user is not dragging inside the specified div element
childDiv[0].addEventListener("dragleave", ()=>{
	childDiv[0].classList.remove("active");
  dragText1.textContent = "Drag & Drop to Upload File";
});

childDiv[1].addEventListener("dragleave", ()=>{
	childDiv[1].classList.remove("active");
	dragText2.textContent = "Drag & Drop to Upload File";
});

childDiv[2].addEventListener("dragleave", ()=>{
	childDiv[2].classList.remove("active");
	dragText3.textContent = "Drag & Drop to Upload File";
});

//Allows a user to drop a file inside the specified div
childDiv[0].addEventListener("drop", (event)=>{
	event.preventDefault();
	file1 = event.dataTransfer.files[0];
	if(file1){
		const reader = new FileReader();
		reader.onload = function(){
		const result = reader.result;
		img1.src = result;
		imgDiv1.setAttribute('style', 'display:block !important');
		childDiv[0].classList.add("active");
		}
		cancelBtn1.addEventListener("click", function(){
			img1.src = "";
			imgDiv1.setAttribute('style', 'display:none !important');
			childDiv[0].classList.remove("active");
		})
		reader.readAsDataURL(file1);
	}
	if(this.value){
		let valueStore = this.value.match(regExp);
		fileName1.textContent = valueStore;
	}
});

childDiv[1].addEventListener("drop", (event)=>{
	event.preventDefault();
	file2 = event.dataTransfer.files[0];
	showFile();
});

childDiv[2].addEventListener("drop", (event)=>{
	event.preventDefault();
	file3 = event.dataTransfer.files[0];
	showFile();
});