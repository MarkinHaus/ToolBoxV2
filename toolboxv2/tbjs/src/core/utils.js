// tbjs/core/utils.js
// Generic utility functions.
// Original: autocomplete from util.js, other small helpers scattered.

import logger from './logger.js';

/**
 * Basic autocomplete functionality.
 * @param {HTMLInputElement} inp - The input field element.
 * @param {Array<string>} arr - An array of possible autocompleted values.
 * Original: web/scripts/util.js
 */
export function autocomplete(inp, arr) {
    // ... (Your existing autocomplete code from util.js)
    // Ensure it doesn't have hardcoded class names if possible, or make them configurable.
    // Consider if this should be a UI component: TB.ui.Autocomplete(inputEl, options)
    logger.debug('[Utils] Autocomplete attached to input:', inp);
     /*the autocomplete function takes two arguments,
  the text field element and an array of possible autocompleted values:*/
  var currentFocus;
  /*execute a function when someone writes in the text field:*/
  inp.addEventListener("input", function(e) {
      var a, b, i, val = this.value;
      /*close any already open lists of autocompleted values*/
      closeAllLists();
      if (!val) { return false;}
      currentFocus = -1;
      /*create a DIV element that will contain the items (values):*/
      a = document.createElement("DIV");
      a.setAttribute("id", this.id + "autocomplete-list");
      a.setAttribute("class", "autocomplete-items"); // TODO: Make class configurable or use Tailwind
      /*append the DIV element as a child of the autocomplete container:*/
      this.parentNode.appendChild(a);
      /*for each item in the array...*/
      for (i = 0; i < arr.length; i++) {
        /*check if the item starts with the same letters as the text field value:*/
        if (arr[i].substr(0, val.length).toUpperCase() == val.toUpperCase()) {
          /*create a DIV element for each matching element:*/
          b = document.createElement("DIV");
          /*make the matching letters bold:*/
          b.innerHTML = "<strong>" + arr[i].substr(0, val.length) + "</strong>";
          b.innerHTML += arr[i].substr(val.length);
          /*insert a input field that will hold the current array item's value:*/
          b.innerHTML += "<input type='hidden' value='" + arr[i] + "'>";
          /*execute a function when someone clicks on the item value (DIV element):*/
              b.addEventListener("click", function(e) {
              /*insert the value for the autocomplete text field:*/
              inp.value = this.getElementsByTagName("input")[0].value;
              /*close the list of autocompleted values,
              (or any other open lists of autocompleted values:*/
              closeAllLists();
          });
          a.appendChild(b);
        }
      }
  });
  /*execute a function presses a key on the keyboard:*/
  inp.addEventListener("keydown", function(e) {
      var x = document.getElementById(this.id + "autocomplete-list");
      if (x) x = x.getElementsByTagName("div");
      if (e.keyCode == 40) {
        currentFocus++;
        addActive(x);
      } else if (e.keyCode == 38) { //up
        currentFocus--;
        addActive(x);
      } else if (e.keyCode == 13) {
        e.preventDefault();
        if (currentFocus > -1) {
          if (x) x[currentFocus].click();
        }
      }
  });
  function addActive(x) {
    if (!x) return false;
    removeActive(x);
    if (currentFocus >= x.length) currentFocus = 0;
    if (currentFocus < 0) currentFocus = (x.length - 1);
    x[currentFocus].classList.add("autocomplete-active"); // TODO: Make class configurable
  }
  function removeActive(x) {
    for (var i = 0; i < x.length; i++) {
      x[i].classList.remove("autocomplete-active");
    }
  }
  function closeAllLists(elmnt) {
    var x = document.getElementsByClassName("autocomplete-items");
    for (var i = 0; i < x.length; i++) {
      if (elmnt != x[i] && elmnt != inp) {
      x[i].parentNode.removeChild(x[i]);
    }
  }
}
document.addEventListener("click", function (e) {
    closeAllLists(e.target);
});
}


/**
 * Debounces a function, ensuring it's only called after a certain delay
 * since the last time it was invoked.
 * @param {Function} func - The function to debounce.
 * @param {number} delay - The debounce delay in milliseconds.
 * @returns {Function} The debounced function.
 */
export function debounce(func, delay) {
    let timeoutId;
    return function(...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => {
            func.apply(this, args);
        }, delay);
    };
}

/**
 * Throttles a function, ensuring it's only called at most once
 * within a specified time window.
 * @param {Function} func - The function to throttle.
 * @param {number} limit - The throttle time window in milliseconds.
 * @returns {Function} The throttled function.
 */
export function throttle(func, limit) {
    let বাক্যInThrottle;
    return function(...args) {
        if (!বাক্যInThrottle) {
            func.apply(this, args);
            বাক্যInThrottle = true;
            setTimeout(() => বাক্যInThrottle = false, limit);
        }
    };
}

/**
 * Generates a simple unique ID.
 * @param {string} [prefix='id-'] - Prefix for the ID.
 * @returns {string} A unique ID string.
 */
export function uniqueId(prefix = 'id-') {
    return prefix + Math.random().toString(36).substr(2, 9);
}

/**
 * Deep clones an object or array.
 * @param {object|Array} obj - The object/array to clone.
 * @returns {object|Array} The cloned object/array.
 */
export function deepClone(obj) {
    if (obj === null || typeof obj !== 'object') {
        return obj;
    }
    // Check if it's a Date object
    if (obj instanceof Date) {
        return new Date(obj.getTime());
    }
    // Check if it's an Array
    if (Array.isArray(obj)) {
        const clonedArray = [];
        for (let i = 0; i < obj.length; i++) {
            clonedArray[i] = deepClone(obj[i]);
        }
        return clonedArray;
    }
    // It's an Object
    const clonedObject = {};
    for (const key in obj) {
        if (Object.prototype.hasOwnProperty.call(obj, key)) {
            clonedObject[key] = deepClone(obj[key]);
        }
    }
    return clonedObject;
}

/**
 * Cleans a URL, removing protocol and potentially multiple slashes.
 * Original: cleanUrl from original index.js
 * @param {string} url - The URL to clean.
 * @returns {string} The cleaned URL.
 */
export function cleanUrl(url) {
    if (!url) return '';
    // Remove protocol
    let cleaned = url.replace(/^(https?:\/\/)/i, '');
    // Normalize slashes (e.g., //path -> /path), but be careful not to break //domain.com
    // This needs to be context-aware; for now, simpler version:
    // cleaned = cleaned.replace(/\/{2,}/g, '/');
    return cleaned;
}


export function escapeHtml (unsafe) {
        if (typeof unsafe !== 'string') return '';
        return unsafe
            .replace(/&/g, "&")
            .replace(/</g, "<")
            .replace(/>/g, ">")
            .replace(/"/g, '"')
            .replace(/'/g, "'");
    };
