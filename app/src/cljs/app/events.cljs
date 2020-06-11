(ns app.events
  (:require
   [re-frame.core :as re-frame]
   [app.db :as db]
   [day8.re-frame.http-fx]
   [ajax.core :as ajax]
   [cljs.pprint :refer [pprint]]))

(re-frame/reg-event-db
 ::initialize-db
 (fn [_ _]
   db/default-db))

;; (re-frame/reg-event-fx
;;  :add-text
;;  (fn [{:keys [db]} [_ text]]
;;    {:db (assoc db :text text)}))

(re-frame/reg-event-db
 :update-text
 (fn [db [_ text]]
   (assoc db :text text)))

(re-frame/reg-event-db
 :emoji-fetch-success
 (fn [db [_ resp]]
   (let [emojis (:predictions resp)]
     (assoc db :emojis emojis))))

(re-frame/reg-event-db
 :emoji-fetch-failure
 (fn [db [_ resp]]
     (assoc db :emoji-fetch-failure resp)))

(re-frame/reg-event-fx
 :fetch-emojis
 (fn [{:keys [db]} [_ text]]
   {:http-xhrio {:method :post
                 :uri "/tunteuta"
                 :params {:text text :num 5}
                 :format (ajax/json-request-format)
                 :response-format (ajax/json-response-format {:keywords? true})
                 :on-success [:emoji-fetch-success]
                 :on-failure [:emoji-fetch-failure]}}))
