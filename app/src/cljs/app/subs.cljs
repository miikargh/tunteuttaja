(ns app.subs
  (:require
   [re-frame.core :as re-frame]))

(re-frame/reg-sub
 ::name
 (fn [db]
   (:name db)))

(re-frame/reg-sub
 ::text
 (fn [db]
   (:text db)))

(re-frame/reg-sub
 ::emoji-fetch-failure
 (fn [db]
   (:emoji-fetch-failure db)))

(re-frame/reg-sub
 ::emojis
 (fn [db]
   (:emojis db)))
