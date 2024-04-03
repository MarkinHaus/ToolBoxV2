var userAgent = navigator.userAgent;
var platform = navigator.platform;

var gecko = /gecko\/\d/i.test(userAgent);
var ie_upto10 = /MSIE \d/.test(userAgent);
var ie_11up = /Trident\/(?:[7-9]|\d{2,})\..*rv:(\d+)/.exec(userAgent);
var edge = /Edge\/(\d+)/.exec(userAgent);
var ie = ie_upto10 || ie_11up || edge;
var ie_version = ie && (ie_upto10 ? document.documentMode || 6 : +(edge || ie_11up)[1]);
var webkit = !edge && /WebKit\//.test(userAgent);
var qtwebkit = webkit && /Qt\/\d+\.\d+/.test(userAgent);
var chrome = !edge && /Chrome\/(\d+)/.exec(userAgent);
var chrome_version = chrome && +chrome[1];
var presto = /Opera\//.test(userAgent);
var safari = /Apple Computer/.test(navigator.vendor);
var mac_geMountainLion = /Mac OS X 1\d\D([8-9]|\d\d)\D/.test(userAgent);
var phantom = /PhantomJS/.test(userAgent);

var ios = safari && (/Mobile\/\w+/.test(userAgent) || navigator.maxTouchPoints > 2);
var android = /Android/.test(userAgent);
// This is woefully incomplete. Suggestions for alternative methods welcome.
var mobile = ios || android || /webOS|BlackBerry|Opera Mini|Opera Mobi|IEMobile/i.test(userAgent);
var mac = ios || /Mac/.test(platform);
var chromeOS = /\bCrOS\b/.test(userAgent);
var windows = /win/i.test(platform);

var presto_version = presto && userAgent.match(/Version\/(\d*\.\d*)/);

console.log("Device Information :\n ",
"\n[userAgent]:",userAgent,
"\n[platform]:",platform,
"\n[gecko]:",gecko,
"\n[ie_upto10]:",ie_upto10,
"\n[ie_11up]:",ie_11up,
"\n[edge]:",edge,
"\n[ie]:",ie,
"\n[ie_version]:",ie_version,
"\n[webkit]:",webkit,
"\n[qtwebkit]:",qtwebkit,
"\n[chrome]:",chrome,
"\n[chrome_version]:",chrome_version,
"\n[presto]:",presto,
"\n[safari]:",safari,
"\n[mac_geMountainLion]:",mac_geMountainLion,
"\n[phantom]:",phantom,
"\n[ios]:",ios,
"\n[android]:",android,
"\n[mobile]:",mobile,
"\n[mac]:",mac,
"\n[chromeOS]:",chromeOS,
"\n[windows]:",windows,
"\n[presto_version]:",presto_version,
)
