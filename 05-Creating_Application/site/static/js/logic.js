function populateDropDown(dropdownID) {
    d3.json(`http://127.0.0.1:8000/api/v1.0/${dropdownID.toLowerCase()}`).then(function(data) {
        let dropDown = d3.select(`#sel${dropdownID}`);

        data.forEach (l => {
            dropDown.append('option').text(l[dropdownID]).property('value', l[`${dropdownID}_ID`]);
        });

        dropDown.property('selectedIndex', -1);
    });
}

function winPct(wins, losses) {
    return wins / (wins + losses);
}

function submitForm() {
    payload = {};

    // home team data
    payload.home_team = d3.select("#selHomeTeams option:checked").text();
    payload.home_win_pct = winPct(parseInt(d3.select("#txtHomeWins").node().value), parseInt(d3.select("#txtHomeLosses").node().value));
    var homeStreak = d3.select('input[name="home_streak"]:checked').property("value"); 
    if (homeStreak == 'W') {
        payload.home_win_streak = d3.select("#txtHomeStreak").node().value;
        payload.home_loss_streak = 0;
    }
    else {
        payload.home_win_streak = 0;
        payload.home_loss_streak = d3.select("#txtHomeStreak").node().value;
    }
    payload.home_avg_pts_for = d3.select("#txtHomePtsFor").node().value;
    payload.home_avg_pts_against = d3.select("#txtHomePtsAgainst").node().value;

    // away team data
    payload.away_team = d3.select("#selAwayTeams option:checked").text();
    payload.away_win_pct = winPct(parseInt(d3.select("#txtAwayWins").node().value), parseInt(d3.select("#txtAwayLosses").node().value));
    var awayStreak = d3.select('input[name="away_streak"]:checked').property("value"); 
    if (awayStreak == 'W') {
        payload.away_win_streak = d3.select("#txtAwayStreak").node().value;
        payload.away_loss_streak = 0;
    }
    else {
        payload.away_win_streak = 0;
        payload.away_loss_streak = d3.select("#txtAwayStreak").node().value;
    }
    payload.away_avg_pts_for = d3.select("#txtAwayPtsFor").node().value;
    payload.away_avg_pts_against = d3.select("#txtAwayPtsAgainst").node().value;

    // event details
    payload.venue = d3.select("#selVenues option:checked").text();
    payload.city = d3.select("#selCities option:checked").text();
    payload.state = d3.select("#selStates option:checked").text();
    payload.month = d3.select("#selMonths option:checked").text();
    payload.week_num = d3.select("#txtWeekNum").node().value;
    payload.time = d3.select("#selTimes option:checked").text();
    payload.weather = d3.select("#selWeather option:checked").text();
    payload.temp = d3.select("#txtTemp").node().value;

    d3.json('http://127.0.0.1:8000/api/v1.0/make_prediction', {
        method: "POST",
        headers: {
            "Content-Type": "application/x-www-form-urlencoded"
        },
        body: "data=" + encodeURIComponent(JSON.stringify(payload))
    }).then(function(data) {
        console.log(data['winning_team']);
        d3.select('#navButtons').style("display", "none");
        d3.select('#navIndicators').style("display", "none");
        d3.select('#divResult').style("display", "block");
        d3.select('#teamLogo').property('src', `/static/img/${data['winning_team']}.png`)
    });
}

const dropDowns = ["Cities", "Months", "States", "Times", "Venues", "Weather", "HomeTeams", "AwayTeams"];
dropDowns.forEach(populateDropDown);

