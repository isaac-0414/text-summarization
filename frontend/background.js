chrome.runtime.onInstalled.addListener(function() {
   chrome.contextMenus.create({
      "id": "summarizeWebpage",
      "title": "Summarize this webpage",
      "contexts": ["all"]
   });
});

chrome.contextMenus.onClicked.addListener(async () => {
   const activeTab = await getActiveTabURL();

   chrome.tabs.sendMessage(activeTab.id, {
      type: "show-modal"
   });
});
   
async function getActiveTabURL() {
   const tabs = await chrome.tabs.query({
         currentWindow: true,
         active: true
   });
   
   return tabs[0];
}