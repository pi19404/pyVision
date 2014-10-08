jQuery.gitUser = function (username, callback) {
	    jQuery.getJSON('https://api.github.com/users/' + username + '/repos?per_page=100&callback=?', callback) //Change per_page according to your need.
	}

	jQuery.fn.getRepos = function (username) {
	    this.html("<h2 style=\"color:#FFF;\">Hold on tight, digging out " + username + "'s repositories...</h2><br>");

	    var target = this;
	    $.gitUser(username, function (data) {
		var repos = data.data; // JSON Parsing
		//alert(repos.length); Only for checking how many items are returned.
		sortByForks(repos); //Sorting by forks. You can customize it according to your needs.
		var list = $('<section>');
		target.empty().append(list);
		count=0;
		fcount=0;
		fwatch=0;
		$(repos).each(function () {
		    checkfork = this.fork;
		    if ((this.name != (username.toLowerCase() + '.github.com')) && (checkfork != true)) { //Check for username.github.com repo and for forked projects
		     count++;  
		     fcount=fcount+this.forks; 
		     fwatch=fwatch+this.watchers;
		        //Similarly fetch everything else you need.
		    }

		});
		list.append(' <a class="button" id="view-on-github" href="https://github.com/pi19404/pyVision"><span>Repo '+count+'</span></a>')
                //list.append(' <a class="button" id="view-on-github" href=""> <span> Forks: ' + fcount  + '</span></a>');
		list.append('</section>');
	    });

	    function sortByForks(repos) {
		repos.sort(function (a, b) {
		    return b.forks - a.forks; //Descending order for number of forks based sorting.
		});
	    }
	};
