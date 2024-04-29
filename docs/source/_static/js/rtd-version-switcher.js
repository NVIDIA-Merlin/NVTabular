var jQuery = (typeof(window) != 'undefined') ? window.jQuery : require('jquery');
var doc = $(document);
doc.on('click', "[data-toggle='rst-current-version']", function() {
    $("[data-toggle='rst-versions']").toggleClass("shift-up");
});
