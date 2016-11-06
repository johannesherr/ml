

var w = 600;
var h = 600;

var svg = d3.select('#root')
    .append('svg')
    .attr('width', w)
    .attr('height', h);

d3.json('samples', function(data) {
    samples = data.samples;

    var x = d3.scaleLinear()
        .range([0, w])
        .domain(d3.extent(samples, d => d.x));

    var y = d3.scaleLinear()
        .range([0, h])
        .domain([d3.max(samples, d => d.y), d3.min(samples, d => d.y)]);

var colors = 'red green blue'.split(/ /);

    svg
        .selectAll('rect')
        .data(samples)
        .enter()
        .append('rect')
        .attr('width', 5)
        .attr('height', 5)
        .attr('x', d => x(d.x))
        .attr('y', d => y(d.y))
        .attr('fill', d => colors[d.type]);

    var line = d3.line()
        .x(d => x(d[0]))
        .y(d => y(d[1]));

    var bounds = data.boundaries;
    var lines = bounds.map(b => [[-10, -10 * b[0] + b[1]],
                                 [10, 10 * b[0] + b[1]]]);

    svg
        .selectAll('path')
        .data(lines)
        .enter()
        .append('path')
        .attr('d', line)
        .attr('stroke', (d,i) => colors[i]);
})


