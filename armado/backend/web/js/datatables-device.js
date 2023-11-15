// Call the dataTables jQuery plugin
$(document).ready(function() {
  $('#dataTable').DataTable({
    "paging": false,
    "searching": false,
    "processing": true,
    "serverSide": true,
    "ajax": {
      "url": `${ajax_url}`,
      "type": "POST"
    }
  });
});
