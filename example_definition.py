class FireHandler:
    fire_cache: dict

    def initialize_fires(
        fires: list[dict],
        maximum_number_of_fires: int,
        **kwargs: dict[str, any]
    ) -> None:
        """ initializes the initial state of each fire linking to their ID.
        
        Args:
            fires: a list of dictionaries, each dictionary containing the current state of the fire.
            kwargs: configuration variables specified in the config that are specific to fire logic.
            maximum_number_of_fires: the maximum size of the fires list.

        Return:
            None
        """
        
        # For each fire in fires:
        #   initialize any state or behaviour of Fire given kwargs
        #   register fire and state with cache
            

    def update_fires(
        fires: list[dict],
    ) -> list[dict]:
        """ an update function which alters the state of the fires.

        Args:
            fires: a list of dictionaries, each dictionary containing the current state of the fire.

        Return:
            fires: a list of dictionaries, each dictionary containing the updated state of the fire.
        """
        
        # for each fire in fires:
        #   check the cache given fire['id'] to get old state.
        #   update fire state in cache.
        #   update fire dictionary in fires list
        #   potentially create new fires.
        # return fires
        

# each dictionary within the list of fires input argument could look like this:
fire = {
    "id": 0,                            # some way of uniquely identifying the fire, which is consistent between 'steps'
    "radius": 0.2,                      # the radius of the fire
    "current_number_of_touches": 3,     # the number of touches from agents during the current turn.
    "touch_threshold": 10,              # the maximum number of touches the fire can take before being 'put out'
}

# within the fire cache, we may have more information about each fire:
fire_cache = {
    '0': {
        "radius": 0.2,
        "intensity": 1000,
        "growth_chance": 0.5,
    }
}