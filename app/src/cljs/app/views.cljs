(ns app.views
  (:require
   [re-frame.core :as re-frame]
   [reagent.core :as reagent :refer [atom]]
   [app.subs :as subs]
   [cljs.pprint :refer [pprint]]))

(defn emoji-list []
  (let [emojis (re-frame/subscribe [::subs/emojis])]
    (fn []
      [:div
       (into [:ul.emoji-list] (map #(vector :li [:img.emoji {:src (str "/app/emojis/" (:unicode %) ".svg")
                                                             :alt (str (:alt %))}]) @emojis))])))

(defn text-field []
  (let [text (atom @(re-frame/subscribe [::subs/text]))
        errors (re-frame/subscribe [::subs/emoji-fetch-failure])
        emojis (re-frame/subscribe [::subs/emojis])]
    (fn []
      [:div
       [:textarea.textarea {:placeholder (when (not @text) "Kirjoita tähän jotain ja emojisoi!")
                            :value (str @text)
                            :on-change #(let [value (-> % .-target .-value)]
                                          (reset! text value)
                                          (re-frame/dispatch [:update-text value]))}]
       [:div
        [:button.button.submit-btn.is-primary {:type "submit"
                                               :on-click #(re-frame/dispatch [:fetch-emojis @text])}
         "Emojify!"]
        [emoji-list]]])))

(defn main-panel []
  [:section.section
   [:div.container
    [:h1.title "Tunteuttaja"
     [text-field]]]])
