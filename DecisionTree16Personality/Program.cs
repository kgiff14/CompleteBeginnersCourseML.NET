using DecisionTree16Personality;
using DecisionTree16Personality.AppLogic;
using DecisionTree16Personality.AppLogic.Abstracts;
using DecisionTree16Personality.AppLogic.Implementations;
using DecisionTree16Personality.Models;
using Microsoft.ML.Trainers;

var predictor = new Predictor();

var newSample = new PersonalityInput
{
    Question1 = 0f,
    Question2 = 0f,
    Question3 = 0f,
    Question4 = 0f,
    Question5 = 0f,
    Question6 = 1f,
    Question7 = 1f,
    Question8 = 0f,
    Question9 = 0f,
    Question10 = 0f,
    Question11 = 0f,
    Question12 = 1f,
    Question13 = 1f,
    Question14 = 0f,
    Question15 = 1f,
    Question16 = -1f,
    Question17 = -1f,
    Question18 = 0f,
    Question19 = 0f,
    Question20 = 0f,
    Question21 = 0f,
    Question22 = 0f,
    Question23 = 0f,
    Question24 = 0f,
    Question25 = 0f,
    Question26 = 0f,
    Question27 = 0f,
    Question28 = 0f,
    Question29 = 0f,
    Question30 = -1f,
    Question31 = 0f,
    Question32 = 0f,
    Question33 = 0f,
    Question34 = 0f,
    Question35 = 0f,
    Question36 = -1f,
    Question37 = 0f,
    Question38 = -1f,
    Question39 = 1f,
    Question40 = -1f,
    Question41 = 0f,
    Question42 = 0f,
    Question43 = 1f,
    Question44 = 0f,
    Question45 = -1f,
    Question46 = 0f,
    Question47 = 0f,
    Question48 = 0f,
    Question49 = 0f,
    Question50 = 0f,
    Question51 = 0f,
    Question52 = 0f,
    Question53 = 0f,
    Question54 = 0f,
    Question55 = -1f,
    Question56 = 0f,
    Question57 = 0f,
    Question58 = 0f,
    Question59 = 0f,
    Question60 = 0f
};

Console.WriteLine("\n---------------------------------------Light Gbm Classification Trainers-------------------------------------------------");

var maxEntropyTrainers = new List<TrainerAbstract<OneVersusAllModelParameters>>
{
    new LightGbmMultiClassificationTrainer()
};

maxEntropyTrainers.ForEach(x => predictor.Predict(newSample, x));

Console.WriteLine("\n-----------------AutoML------------------\n");

//Load sample data
var sampleData = new _16Personality.ModelInput()
{
    Response_Id = 1F,
    You_regularly_make_new_friends_ = 0F,
    You_spend_a_lot_of_your_free_time_exploring_various_random_topics_that_pique_your_interest = 0F,
    Seeing_other_people_cry_can_easily_make_you_feel_like_you_want_to_cry_too = -2F,
    You_often_make_a_backup_plan_for_a_backup_plan_ = -3F,
    You_usually_stay_calm__even_under_a_lot_of_pressure = -1F,
    At_social_events__you_rarely_try_to_introduce_yourself_to_new_people_and_mostly_talk_to_the_ones_you_already_know = 2F,
    You_prefer_to_completely_finish_one_project_before_starting_another_ = -2F,
    You_are_very_sentimental_ = 0F,
    You_like_to_use_organizing_tools_like_schedules_and_lists_ = 3F,
    Even_a_small_mistake_can_cause_you_to_doubt_your_overall_abilities_and_knowledge_ = 0F,
    You_feel_comfortable_just_walking_up_to_someone_you_find_interesting_and_striking_up_a_conversation_ = -2F,
    You_are_not_too_interested_in_discussing_various_interpretations_and_analyses_of_creative_works_ = 0F,
    You_are_more_inclined_to_follow_your_head_than_your_heart_ = -2F,
    You_usually_prefer_just_doing_what_you_feel_like_at_any_given_moment_instead_of_planning_a_particular_daily_routine_ = 1F,
    You_rarely_worry_about_whether_you_make_a_good_impression_on_people_you_meet_ = 1F,
    You_enjoy_participating_in_group_activities_ = -2F,
    You_like_books_and_movies_that_make_you_come_up_with_your_own_interpretation_of_the_ending_ = -2F,
    Your_happiness_comes_more_from_helping_others_accomplish_things_than_your_own_accomplishments_ = 1F,
    You_are_interested_in_so_many_things_that_you_find_it_difficult_to_choose_what_to_try_next_ = 0F,
    You_are_prone_to_worrying_that_things_will_take_a_turn_for_the_worse_ = 3F,
    You_avoid_leadership_roles_in_group_settings_ = 1F,
    You_are_definitely_not_an_artistic_type_of_person_ = 2F,
    You_think_the_world_would_be_a_better_place_if_people_relied_more_on_rationality_and_less_on_their_feelings_ = 0F,
    You_prefer_to_do_your_chores_before_allowing_yourself_to_relax_ = 0F,
    You_enjoy_watching_people_argue_ = 1F,
    You_tend_to_avoid_drawing_attention_to_yourself_ = -2F,
    Your_mood_can_change_very_quickly_ = -2F,
    You_lose_patience_with_people_who_are_not_as_efficient_as_you_ = 0F,
    You_often_end_up_doing_things_at_the_last_possible_moment_ = -2F,
    You_have_always_been_fascinated_by_the_question_of_what__if_anything__happens_after_death_ = 1F,
    You_usually_prefer_to_be_around_others_rather_than_on_your_own_ = 2F,
    You_become_bored_or_lose_interest_when_the_discussion_gets_highly_theoretical_ = 0F,
    You_find_it_easy_to_empathize_with_a_person_whose_experiences_are_very_different_from_yours_ = 0F,
    You_usually_postpone_finalizing_decisions_for_as_long_as_possible_ = 0F,
    You_rarely_second_guess_the_choices_that_you_have_made_ = -1F,
    After_a_long_and_exhausting_week__a_lively_social_event_is_just_what_you_need_ = -1F,
    You_enjoy_going_to_art_museums_ = 1F,
    You_often_have_a_hard_time_understanding_other_people_s_feelings_ = 2F,
    You_like_to_have_a_to_do_list_for_each_day_ = 1F,
    You_rarely_feel_insecure_ = -1F,
    You_avoid_making_phone_calls_ = -1F,
    You_often_spend_a_lot_of_time_trying_to_understand_views_that_are_very_different_from_your_own_ = 2F,
    In_your_social_circle__you_are_often_the_one_who_contacts_your_friends_and_initiates_activities_ = -1F,
    If_your_plans_are_interrupted__your_top_priority_is_to_get_back_on_track_as_soon_as_possible_ = 1F,
    You_are_still_bothered_by_mistakes_that_you_made_a_long_time_ago_ = 2F,
    You_rarely_contemplate_the_reasons_for_human_existence_or_the_meaning_of_life_ = 0F,
    Your_emotions_control_you_more_than_you_control_them_ = 1F,
    You_take_great_care_not_to_make_people_look_bad__even_when_it_is_completely_their_fault_ = 0F,
    Your_personal_work_style_is_closer_to_spontaneous_bursts_of_energy_than_organized_and_consistent_efforts_ = 1F,
    When_someone_thinks_highly_of_you__you_wonder_how_long_it_will_take_them_to_feel_disappointed_in_you_ = 0F,
    You_would_love_a_job_that_requires_you_to_work_alone_most_of_the_time_ = 0F,
    You_believe_that_pondering_abstract_philosophical_questions_is_a_waste_of_time_ = 0F,
    You_feel_more_drawn_to_places_with_busy__bustling_atmospheres_than_quiet__intimate_places_ = -2F,
    You_know_at_first_glance_how_someone_is_feeling_ = 0F,
    You_often_feel_overwhelmed_ = 2F,
    You_complete_things_methodically_without_skipping_over_any_steps_ = 0F,
    You_are_very_intrigued_by_things_labeled_as_controversial_ = -1F,
    You_would_pass_along_a_good_opportunity_if_you_thought_someone_else_needed_it_more_ = -1F,
    You_struggle_with_deadlines_ = -1F,
    You_feel_confident_that_things_will_work_out_for_you_ = 3F,
};

//Load model and predict output
var result = _16Personality.Predict(sampleData);

Console.WriteLine($"Personality: {result.PredictedLabel}");