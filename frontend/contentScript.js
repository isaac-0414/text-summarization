const serverURL = "http://127.0.0.1:8000/";

chrome.runtime.onMessage.addListener((obj, sender, response) => {
   const { type } = obj;

   if (type === "show-modal") {
      console.log("show modal")
      function showModal(titleHtml, contentHtml, buttons) {
         const modal = document.createElement("div");
         modal.classList.add("modal_isaac");
         modal.innerHTML = `
               <div class="modal_isaac__inner">
                  <div class="modal_isaac__top">
                     <div class="modal_isaac__title">${titleHtml}</div>
                     <button class="modal_isaac__close" type="button">
                           <span class="material-icons">close</span>
                     </button>
                  </div>
                  <div class="modal_isaac__content">${contentHtml}</div>
                  <div class="modal_isaac__bottom"><span class="copied-msg" hidden>Copied!</span></div>
               </div>
         `;
      
         for (const button of buttons) {
         const element = document.createElement("button");
      
         element.setAttribute("type", "button");
         element.classList.add("modal_isaac__button");
         element.textContent = button.label;
         element.addEventListener("click", () => {
            if (button.triggerClose) {
               document.body.removeChild(modal);
            }
      
            button.onClick(modal);
         });
      
         modal.querySelector(".modal_isaac__bottom").appendChild(element);
         }
      
         modal.querySelector(".modal_isaac__close").addEventListener("click", () => {
         document.body.removeChild(modal);
         });
      
         document.body.appendChild(modal);
      }
      
      showModal("Summary of this webpage", 
      `<textarea name="summarization" id="summarization" cols="30" rows="10" placeholder="Please wait for a while, summarization takes some time..."></textarea>`, 
      [
         {
            label: "Copy",
            onClick: (modal) => {
               const summarizationArea = document.querySelector('.modal_isaac__content #summarization');
               navigator.clipboard.writeText(summarizationArea.value);
               document.querySelector(".copied-msg").hidden = false;
            },
            triggerClose: false
         }
      ]);   
      summarize(document.querySelector('body').innerText);
   }
   response();
});


async function summarize(textToSummarize) {
   const summarizationArea = document.querySelector('.modal_isaac__content #summarization');
   if (textToSummarize.length === 0) {
      summarizationArea.placeholder = "Please input something for summarization";
      return;
   }
   let refreshIntervalId = loadingAnimation();
   const { summary, error } = await callSummarizationAPI(textToSummarize);
   clearInterval(refreshIntervalId);
   if (error) {
      summarizationArea.value = error;
   } else {
      summarizationArea.value = summary;
   }
}

async function callSummarizationAPI(text) {
   try {
      const res = await fetch(serverURL + "summarize/", {
         method: 'POST',
         headers: {
            'Content-Type': 'application/json'
         },
         body: JSON.stringify ({
            "text": text
         })
      })
      console.log(res);
      return await res.json();
   } catch (error) {
      console.error(error);
      return {error: "failed to connect to server"};
   }
}

function loadingAnimation() {
   const summarizationArea = document.querySelector('.modal_isaac__content #summarization');
   function dotsAnimation() {
      summarizationArea.placeholder = "Please wait for a while, summarization takes some time";
      setTimeout(() => {
         summarizationArea.placeholder = "Please wait for a while, summarization takes some time.";
         setTimeout(() => {
            summarizationArea.placeholder = "Please wait for a while, summarization takes some time..";
            setTimeout(() => {
               summarizationArea.placeholder = "Please wait for a while, summarization takes some time...";
            }, 500)
         }, 500)
      }, 500)
   }
   let refreshTimeoutId = dotsAnimation();
   let refreshIntervalId = setInterval(() => {
      dotsAnimation();
   }, 2000)
   return refreshTimeoutId, refreshIntervalId;
}