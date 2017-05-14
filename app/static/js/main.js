$(document).ready(function(){

	$('#submit-form').on('submit', function(e) { //use on if jQuery 1.7+
        e.preventDefault();  //prevent form from submitting
        var data = $("#submit-form :input").serializeArray();
        console.log(data); //use the console for debugging, F12 in Chrome, not alerts

		x=$("#input-url").val();//x is file_name
		if (x==null || x==""){
			alert("please drag an image into the box");
			return false;
		} else {
			feature=$("#input_feature").val();
			request_url="/api/predict?file_name="+x+"&feature="+feature;
			console.log(request_url)
			$.ajax({url: request_url, success: function(result){
				console.log(result)
				$("#prediction_container").html(result);
			}});
		}
    });
	
	$("#input-url").attr("value","");
	
	var $gallery = $( "#gallery" ),
		$searchWindow = $( "#search-window" );

	var $targetedIndex = -1;

	$( "input:submit", "#submit-form" ).button();
	
	$( "div > img", $gallery ).draggable({
		cancel: "a.ui-icon", // clicking an icon won't initiate dragging
		revert: "invalid", // when not dropped, the item will revert back to its initial position
		containment: $( "#demo-frame" ).length ? "#demo-frame" : "document", // stick to demo-frame if present
		helper: "clone",
		cursor: "move"
	});
	
	$searchWindow.droppable({
		accept: "#gallery > div > img",
		activeClass: "ui-state-highlight",
		drop: function(event, ui) {
			targetItem(ui.draggable);
		}
	});	
	
	function targetItem($item) {
		
		var $searchWindowElement = $("div", $searchWindow);
		
		var $oldItem;
		var $oldIndex = -1;
		if($targetedIndex>-1){
			$oldItem = $searchWindowElement.children(":first-child");
			$oldIndex = $targetedIndex+1;
		}
		
		// get the item's index
		$targetedIndex = $item.parent().prevAll().length
		
		// prevent the item from shifting
		$item.parent().css("width", $item.width());
		console.log($item.attr("file_name"))
		
		// update the url
		$("#input-url").attr("value", $item.attr("file_name"));


		$item.fadeOut(function() {
			
			$searchWindowElement.empty();

			$item.css("visibility","hidden")
			$item.appendTo($searchWindowElement);
				var $origHeight = parseFloat($item.height());
				var $origWidth = parseFloat($item.width());
				var $height = 0;
				var $width = 0;
				if($origHeight>$origWidth){
					$height = 200;
					$width = $height*$origWidth/$origHeight;
				}else{
					$width = 200;
					$height = $width*$origHeight/$origWidth;
				}
				$item.css("height", $height + "px");
				$item.width("width", $width + "px");
				$item.css("margin-top", (300-$height)/2 +  "px");	
				$item.css("visibility","visible")
			$searchWindowElement.empty();

			$item.appendTo($searchWindowElement).fadeIn(function(){
				if($oldIndex>0){
					$oldItem.css("height", "96px");
					$oldItem.css("width", "");
					$oldItem.css("margin-top", "");
					$oldItem.appendTo($gallery.children(":nth-child(" + $oldIndex +")")).fadeIn(function(){
						$oldItem.draggable({
							cancel: "a.ui-icon", // clicking an icon won't initiate dragging
							revert: "invalid", // when not dropped, the item will revert back to its initial position
							containment: $( "#demo-frame" ).length ? "#demo-frame" : "document", // stick to demo-frame if present
							helper: "clone",
							cursor: "move"
						});
					});
				}
			});
			
		});

	}	
	
	
	$( "ul.gallery > li" ).click(function( event ) {
		var $item = $( this ),
			$target = $( event.target );
		if ( $target.is( "a.ui-icon-zoomin" ) ) {
			viewLargerImage( $target );
		} else if ( $target.is( "a.ui-icon-refresh" ) ) {
			recycleImage( $item );
		}
		return false;
	});
	
	
});