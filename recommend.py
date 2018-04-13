"""A Yelp-powered Restaurant Recommendation Program"""

from abstractions import *
from data import ALL_RESTAURANTS, CATEGORIES, USER_FILES, load_user_file
from ucb import main, trace, interact
from utils import distance, mean, zip, enumerate, sample, key_of_min_value
from visualize import draw_map
from operator import itemgetter

##################################
# Phase 2: Unsupervised Learning #
##################################


def find_closest(location, centroids):
    """Return the centroid in centroids that is closest to location.
    If multiple centroids are equally close, return the first one.

    >>> find_closest([3.0, 4.0], [[0.0, 0.0], [2.0, 3.0], [4.0, 3.0], [5.0, 5.0]])
    [2.0, 3.0]
    """
    # BEGIN Question 3
    distances = [distance(location,centroid) for centroid in centroids]
    index_of_min = distances.index(min(distances))
    return centroids[index_of_min]
    # END Question 3


def group_by_first(pairs):
    """Return a list of pairs that relates each unique key in the [key, value]
    pairs to a list of all values that appear paired with that key.

    Arguments:
    pairs -- a sequence of pairs

    >>> example = [ [1, 2], [3, 2], [2, 4], [1, 3], [3, 1], [1, 2] ]
    >>> group_by_first(example)
    [[2, 3, 2], [2, 1], [4]]
    """
    keys = []
    for key, _ in pairs:
        if key not in keys:
            keys.append(key)
    return [[y for x, y in pairs if x == key] for key in keys]


def group_by_centroid(restaurants, centroids):
    """Return a list of clusters, where each cluster contains all restaurants
    nearest to a corresponding centroid in centroids. Each item in
    restaurants should appear once in the result, along with the other
    restaurants closest to the same centroid.
    """
    # BEGIN Question 4
    # restuarent is an abstraction
    # compare each lat and long pair to clusters using find_centroid and make lists of closest centroid and restaurant
    restaurant_closest_centroid = [[find_closest(restaurant_location(restuarent),centroids),restuarent] for restuarent in restaurants]
    # you should have a list of centroids
    # call group_by_first and return the result - [[restuaren1,restuarent5,restaurent7],[restuarent3,...],[...]]
    return group_by_first(restaurant_closest_centroid)
    # END Question 4


def find_centroid(cluster):
    """Return the centroid of the locations of the restaurants in cluster."""
    # BEGIN Question 5
    # cluster is a list of restuarents, so call restaurant_location on each element of the list
    list_of_locations = [restaurant_location(restaurant) for restaurant in cluster]
    # now we have a list of lists of lat, long pairs
    # construct a list of latitudes
    list_of_lats = [location[0] for location in list_of_locations]
    # construct a list of longitude
    list_of_longs = [location[1] for location in list_of_locations]
    # call mean on both lists
    # return a list with a average lat and average lon
    return [mean(list_of_lats),mean(list_of_longs)]
    # END Question 5


def k_means(restaurants, k, max_updates=100):
    """Use k-means to group restaurants by location into k clusters."""
    assert len(restaurants) >= k, 'Not enough restaurants to cluster'
    old_centroids, n = [], 0
    # Select initial centroids randomly by choosing k different restaurants
    centroids = [restaurant_location(r) for r in sample(restaurants, k)]

    while old_centroids != centroids and n < max_updates:
        old_centroids = centroids
        # BEGIN Question 6
        # group centroids into clusters
        clusters = group_by_centroid(restaurants,centroids)
        # rebind centroids to new list of cluster centroids
        centroids = [find_centroid(cluster) for cluster in clusters]
        # END Question 6
        n += 1
    return centroids


################################
# Phase 3: Supervised Learning #
################################


def find_predictor(user, restaurants, feature_fn):
    """Return a rating predictor (a function from restaurants to ratings),
    for a user by performing least-squares linear regression using feature_fn
    on the items in restaurants. Also, return the R^2 value of this model.

    Arguments:
    user -- A user
    restaurants -- A sequence of restaurants
    feature_fn -- A function that takes a restaurant and returns a number
    """
    reviews_by_user = {review_restaurant_name(review): review_rating(review)
                       for review in user_reviews(user).values()}

    xs = [feature_fn(r) for r in restaurants]
    ys = [reviews_by_user[restaurant_name(r)] for r in restaurants]

    # BEGIN Question 7

    # compute mean of x values and y values
    x_bar = mean(xs)
    y_bar = mean(ys)
    # sum of (x_i - x_bar)^2 and sum of (y_i - y_bar)^2
    S_x_x = sum([(x_i - x_bar)**2 for x_i in xs])
    S_y_y = sum([(y_i - y_bar)**2 for y_i in ys])
    x_y_pairs = zip([x_i - x_bar for x_i in xs],[y_i - y_bar for y_i in ys])
    S_x_y = sum([pair[0]*pair[1] for pair in x_y_pairs])
    # compute regression coefficients
    b = S_x_y / S_x_x
    a = y_bar - b*x_bar
    r_squared = S_x_y**2 / (S_x_x*S_y_y)
    # END Question 7

    def predictor(restaurant):
        return b * feature_fn(restaurant) + a

    return predictor, r_squared


def best_predictor(user, restaurants, feature_fns):
    """Find the feature within feature_fns that gives the highest R^2 value
    for predicting ratings by the user; return a predictor using that feature.

    Arguments:
    user -- A user
    restaurants -- A list of restaurants
    feature_fns -- A sequence of functions that each takes a restaurant
    """
    reviewed = user_reviewed_restaurants(user, restaurants)
    # BEGIN Question 8
    # for each feature_fn in feature_fns call find_predictor
    r_squared_of_predictor_fns = [find_predictor(user,reviewed,feature_fn)[1] for feature_fn in feature_fns]
    best_predictor_fn_index = r_squared_of_predictor_fns.index(max(r_squared_of_predictor_fns))
    best_feature_fn = feature_fns[best_predictor_fn_index]
    return find_predictor(user, reviewed, best_feature_fn)[0]
    # END Question 8


def rate_all(user, restaurants, feature_fns):
    """Return the predicted ratings of restaurants by user using the best
    predictor based on a function from feature_fns.

    Arguments:
    user -- A user
    restaurants -- A list of restaurants
    feature_fns -- A sequence of feature functions
    """
    predictor = best_predictor(user, ALL_RESTAURANTS, feature_fns)
    reviewed = user_reviewed_restaurants(user, restaurants)
    # BEGIN Question 9
    # for any restaurant that has been reviewed just use that rating, otherwise use best_predictor function
    previously_reviewed = {restaurant_name(restaurant):user_rating(user,restaurant_name(restaurant)) for restaurant in user_reviewed_restaurants(user,restaurants)}
    # now we have a dict of names and rating pairs for user reviewed restaurants
    # now predict the ratings for the restaurents not yet reviewed by this user
    predicted_reviews = {restaurant_name(restaurant):predictor(restaurant) for restaurant in restaurants if restaurant_name(restaurant) not in previously_reviewed}
    # merge lists
    final_dict = dict(previously_reviewed,**predicted_reviews)
    return final_dict
    # END Question 9


def search(query, restaurants):
    """Return each restaurant in restaurants that has query as a category.

    Arguments:
    query -- A string
    restaurants -- A sequence of restaurants
    """
    # BEGIN Question 10
    restaurants_of_category  = [restaurant for restaurant in restaurants if query in restaurant_categories(restaurant)]
    return restaurants_of_category
    # END Question 10


def feature_set():
    """Return a sequence of feature functions."""
    return [lambda r: mean(restaurant_ratings(r)),
            restaurant_price,
            lambda r: len(restaurant_ratings(r)),
            lambda r: restaurant_location(r)[0],
            lambda r: restaurant_location(r)[1]]


@main
def main(*args):
    import argparse
    parser = argparse.ArgumentParser(
        description='Run Recommendations',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-u', '--user', type=str, choices=USER_FILES,
                        default='test_user',
                        metavar='USER',
                        help='user file, e.g.\n' +
                        '{{{}}}'.format(','.join(sample(USER_FILES, 3))))
    parser.add_argument('-k', '--k', type=int, help='for k-means')
    parser.add_argument('-q', '--query', choices=CATEGORIES,
                        metavar='QUERY',
                        help='search for restaurants by category e.g.\n'
                        '{{{}}}'.format(','.join(sample(CATEGORIES, 3))))
    parser.add_argument('-p', '--predict', action='store_true',
                        help='predict ratings for all restaurants')
    parser.add_argument('-r', '--restaurants', action='store_true',
                        help='outputs a list of restaurant names')
    args = parser.parse_args()

    # Output a list of restaurant names
    if args.restaurants:
        print('Restaurant names:')
        for restaurant in sorted(ALL_RESTAURANTS, key=restaurant_name):
            print(repr(restaurant_name(restaurant)))
        exit(0)

    # Select restaurants using a category query
    if args.query:
        restaurants = search(args.query, ALL_RESTAURANTS)
    else:
        restaurants = ALL_RESTAURANTS

    # Load a user
    assert args.user, 'A --user is required to draw a map'
    user = load_user_file('{}.dat'.format(args.user))

    # Collect ratings
    if args.predict:
        ratings = rate_all(user, restaurants, feature_set())
    else:
        restaurants = user_reviewed_restaurants(user, restaurants)
        names = [restaurant_name(r) for r in restaurants]
        ratings = {name: user_rating(user, name) for name in names}

    # Draw the visualization
    if args.k:
        centroids = k_means(restaurants, min(args.k, len(restaurants)))
    else:
        centroids = [restaurant_location(r) for r in restaurants]
    draw_map(centroids, restaurants, ratings)
